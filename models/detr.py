# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR model and criterion classes.
"""
import torch
import torch.nn.functional as F
from torch import nn

from util import box_ops
from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized)

from .backbone import build_backbone
from .matcher import build_matcher
from .segmentation import (DETRsegm, PostProcessPanoptic, PostProcessSegm,
                           dice_loss, sigmoid_focal_loss)
from .transformer import build_transformer


class DETR(nn.Module):
    """ This is the DETR module that performs object detection """
    def __init__(self, backbone, transformer, num_classes, num_queries, aux_loss=False):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.class_embed = nn.Linear(hidden_dim, num_classes + 1)  #num_classes　不包含背景　class_embed生成预测的分类结果，最后一维对应物体类别数量
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3) #　bbox_embed生成预测的回归结果　经过MLP中间每层的维度被映射到hidden_dim，
        # 最后一层维度映射到４，代表bbox的中心店横，纵坐标和宽高．
        self.query_embed = nn.Embedding(num_queries, hidden_dim) #transformer decoder部分的query embedding　初始化query以及对其编码生成嵌入
        # num_queries代表图像中有多少个目标　默认是100个，对这些目标（位置）全部进行嵌入，
        # 维度映射到 hidden_dim，将query_embedding 的权重作为参数输入到Transformer的前向过程，使用时与position encoding的方式相同：直接相加
        self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1) #将CNN提取的特征维度映射到transformer隐层的维度，转化为序列
        self.backbone = backbone
        self.aux_loss = aux_loss  #表示是否要对transformer中的Decoder的每层输出都计算loss

    def forward(self, samples: NestedTensor):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels

            It returns a dict with the following elements:
               - "pred_logits": the classification logits (including no-object) for all queries.
                                Shape= [batch_size x num_queries x (num_classes + 1)]
               - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                               (center_x, center_y, height, width). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized bounding box.
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
        """
        # 将输入样本转换为NestedTensor类型
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)
        features, pos = self.backbone(samples) #输入到CNN提取特征
        # 取出最后一层特征及其mask
        src, mask = features[-1].decompose()
        assert mask is not None
        # Transformer的输出是元组，分别为Decoder和Encoder的输出，因此这里取第一个代表的是Decoder的输出
        hs = self.transformer(self.input_proj(src), mask, self.query_embed.weight, pos[-1])[0]
        #生成分类与回归的预测结果
        outputs_class = self.class_embed(hs)
        outputs_coord = self.bbox_embed(hs).sigmoid()
        #由于hs包含了Transformer中Decoder每层输出，因此索引为-1代表取出最后一层的输出
        out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}
        #　若指定要计算Decoder每层预测输出对应的loss, 则记录对应的输出结果
        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord)
        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a, 'pred_boxes': b}
                for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]


class SetCriterion(nn.Module):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    def __init__(self, num_classes, matcher, weight_dict, eos_coef, losses):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.num_classes = num_classes #类别墅，不包含背景
        self.matcher = matcher #将预测与GT进行匹配的算法
        self.weight_dict = weight_dict #各种loss对应的权重
        self.eos_coef = eos_coef #针对背景分类的loss权重
        self.losses = losses #指定需要计算那些losses  losses = ['labels', 'boxes', 'cardinality','masks']
        empty_weight = torch.ones(self.num_classes + 1) #设置在分类loss中，前景的权重为1，背景的权重由传进来的参数指定
        empty_weight[-1] = self.eos_coef
        self.register_buffer('empty_weight', empty_weight) #将这部分注册到buffer，能够被state_dict记录同事不会有梯度传播到此处

    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        """Classification loss (NLL) 实际调用的是CE LOSS
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']
        #idx是一个tuple,第一个元素是各个object的batch index, 第二个元素是各个object的query index, shape
        # 均为(num_matched_queries1+num_matched_queries2+...)
        idx = self._get_src_permutation_idx(indices) #该方法返回一个tuple，代表所有匹配的预测结果的batch index
        # (在当前batch中属于第几张图像)和 query index(图像中的第几个query对象)
        # 获得当前batch中所有匹配的GT所述的类别(target_classes_o) 通过 src_logits、target_classes_o 设置预测结果对应的GT
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        #(b ,num_queries=100),初始化为背景 target_classes 的shape和 src_logits 一致，代表每个query objects对应的GT
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        #匹配的预测索引对应的值置为匹配的GT
        target_classes[idx] = target_classes_o
        #使用Pytorch的交叉熵损失时，需要将预测类别的那个维度转换到通道这个维度上（dim1）
        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        losses = {'loss_ce': loss_ce}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            #class_error计算的是Top-1精度（百分数），即预测概率最大的那个类别与对应被分配的GT类别是否一致，这部分仅用于log，并不参与模型训练
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes): #仅用作log，不涉及反向传播梯度
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        pred_logits = outputs['pred_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device)
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        #(batch index, queries index) shape都是(num_matched_queries1+num_matched_queries2+...)
        idx = self._get_src_permutation_idx(indices)
        #outputs['pred_boxes']的shape是(b, num_queries=100, 4)
        src_boxes = outputs['pred_boxes'][idx] #src_boxes 的shape是(num_matched_queries1+num_matched_queries2+..., 4)
        #(num_matched_queries1+num_matched_queries2+..., 4)
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')

        losses = {}
        #num_boxes是一个batch图像中目标物体的数量
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes
        # 由于generalized_box_iou返回的是每个预测结果与每个GT的giou,
        #　因此取对角线代表获取的是相互匹配的预测结果与GT的giou
        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes),
            box_ops.box_cxcywh_to_xyxy(target_boxes)))
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        return losses

    def loss_masks(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the masks: the focal loss and the dice loss.
           targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)
        src_masks = outputs["pred_masks"]
        src_masks = src_masks[src_idx]
        masks = [t["masks"] for t in targets]
        # TODO use valid to mask invalid areas due to padding in loss
        target_masks, valid = nested_tensor_from_tensor_list(masks).decompose()
        target_masks = target_masks.to(src_masks)
        target_masks = target_masks[tgt_idx]

        # upsample predictions to the target size
        src_masks = interpolate(src_masks[:, None], size=target_masks.shape[-2:],
                                mode="bilinear", align_corners=False)
        src_masks = src_masks[:, 0].flatten(1)

        target_masks = target_masks.flatten(1)
        target_masks = target_masks.view(src_masks.shape)
        losses = {
            "loss_mask": sigmoid_focal_loss(src_masks, target_masks, num_boxes),
            "loss_dice": dice_loss(src_masks, target_masks, num_boxes),
        }
        return losses

    def _get_src_permutation_idx(self, indices):
        #返回匹配的预测结果的batch index和 queries index
        # permute predictions following indices
        # (num_matched_queries1+num_matched_queries2+...)
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        # (num_matched_queries1+num_matched_queries2+...)
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        #根据指定的loss类型，调用对应的方法来进行计算
        loss_map = {
            'labels': self.loss_labels,
            'cardinality': self.loss_cardinality,
            'boxes': self.loss_boxes,
            'masks': self.loss_masks
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        ''' output是DETR模型的输出，是一个dict，形式如下：
        {'pred_logits':(b,num_queries=100, num_classes), 
         'pred_boxes':(b, num_queries=100, 4),
         'aux_outputs':[{'pred_logits':..., 'pred_boxes':...},{...},...]
        '''
        #过滤掉中间层的输出，只保留最后一层的预测结果
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}

        # Retrieve the matching between the outputs of the last layer and the targets
        # 将网络预测结果与GT匹配，indices是一个包含多个元组的list,长度与batch_size相等，
        # 每个元组为(index_i,index_j)，前者是匹配的预测索引，后者是GT索引
        # 并且len(index_i) = len(index_i) = min(num_queries, num_targets_in_image) 能匹配上的数量取决于图像中的物体数量
        indices = self.matcher(outputs_without_aux, targets)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        # 计算该batch的图像中目标物体的数量，在所有分布式节点之间同步
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            #计算特定类型的loss (这里的loss变量是字符串：'labels', 'boxes', 'cardinality', 'masks', 表示loss类型)
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        # 如果模型输出包含了中间层输出，则一并计算对应的loss
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    if loss == 'masks':
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs = {'log': False}
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses


class PostProcess(nn.Module):
    """ This module converts the model's output into the format expected by the coco api"""
    @torch.no_grad()
    def forward(self, outputs, target_sizes):
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """
        out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']

        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2
        # (b, num_queries=100, num_classes+1)
        prob = F.softmax(out_logits, -1)
        # (b, num_queries=100), (b, num_queries=100)
        scores, labels = prob[..., :-1].max(-1)  #由于coco api中的评估不包含背景类，在生成预测结果时直接排除了背景类，

        # convert to [x0, y0, x1, y1] format
        boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)  #模型在回归部分的输出是归一化的值，需要根据图像尺寸来还原
        # and from relative [0, 1] to absolute [0, height] coordinates
        # (b,), (b,)
        img_h, img_w = target_sizes.unbind(1)
        # (b, 4)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        # (b, num_queries=100, 4) * (b, 1, 4)
        boxes = boxes * scale_fct[:, None, :]

        results = [{'scores': s, 'labels': l, 'boxes': b} for s, l, b in zip(scores, labels, boxes)]

        return results


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            #最后一层生成bbox的c_x,c_Y,w,h,时不需要ReLU激活
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def build(args):
    # the `num_classes` naming here is somewhat misleading.
    # it indeed corresponds to `max_obj_id + 1`, where max_obj_id
    # is the maximum id for a class in your dataset. For example,
    # COCO has a max_obj_id of 90, so we pass `num_classes` to be 91.
    # As another example, for a dataset that has a single class with id 1,
    # you should pass `num_classes` to be 2 (max_obj_id + 1).
    # For more details on this, check the following discussion
    # https://github.com/facebookresearch/detr/issues/108#issuecomment-650269223
    num_classes = 20 if args.dataset_file != 'coco' else 91
    if args.dataset_file == "coco_panoptic":
        # for panoptic, we just add a num_classes that is large enough to hold
        # max_obj_id + 1, but the exact value doesn't really matter
        num_classes = 250
    device = torch.device(args.device)

    backbone = build_backbone(args)

    transformer = build_transformer(args)

    model = DETR(
        backbone,
        transformer,
        num_classes=num_classes,
        num_queries=args.num_queries,
        aux_loss=args.aux_loss,
    )
    if args.masks:
        model = DETRsegm(model, freeze_detr=(args.frozen_weights is not None))
    matcher = build_matcher(args) #需要将网络的预测输出与GT进行配对　这里采用的是匈牙利算法
    weight_dict = {'loss_ce': 1, 'loss_bbox': args.bbox_loss_coef} #各部分loss有着不同的权重稀疏 分别是分类和回归损失
    #分类损失采用的是交叉熵损失，回归损失包括bbox的 L1 Loss（计算x、y、w、h的绝对值误差）与 GIoU Loss
    weight_dict['loss_giou'] = args.giou_loss_coef
    if args.masks:#如果设置masks参数，需要加入赌赢的loss
        weight_dict["loss_mask"] = args.mask_loss_coef
        weight_dict["loss_dice"] = args.dice_loss_coef
    # TODO this is a hack
    if args.aux_loss: #如果为True, 即代表需要计算解码器中间层预测结果对应的loss，那么也要设置对应的loss权重。
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)
    #指定计算哪些类型的loss
    #其中cardinality表示计算预测为前景的数量与GT数量的L1误差,其仅用作log,不涉及反向传播梯度
    losses = ['labels', 'boxes', 'cardinality']
    if args.masks:
        losses += ["masks"]
    #args.eos_coef 用于在计算分类loss中前景和背景的相对权重
    criterion = SetCriterion(num_classes, matcher=matcher, weight_dict=weight_dict,
                             eos_coef=args.eos_coef, losses=losses)
    criterion.to(device)
    postprocessors = {'bbox': PostProcess()}
    if args.masks:
        postprocessors['segm'] = PostProcessSegm()
        if args.dataset_file == "coco_panoptic":
            is_thing_map = {i: i <= 90 for i in range(201)}
            postprocessors["panoptic"] = PostProcessPanoptic(is_thing_map, threshold=0.85)

    return model, criterion, postprocessors
