# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
COCO dataset which returns image_id for evaluation.

Mostly copy-paste from https://github.com/pytorch/vision/blob/13b35ff/references/detection/coco_utils.py
"""
from pathlib import Path

import torch
import torch.utils.data
import torchvision
from pycocotools import mask as coco_mask

import datasets.transforms as T


class CocoDetection(torchvision.datasets.CocoDetection):
    # 该方法首先检查数据文件路径的有效性，然后构造一个字典类型的 PATHS 变量来映射训练集与验证集的路径，
    # 最后实例化一个 CocoDetection() 对象，CocoDetection 这个类继承了torchvision.datasets.CocoDetection
    def __init__(self, img_folder, ann_file, transforms, return_masks):
        super(CocoDetection, self).__init__(img_folder, ann_file)
        self._transforms = transforms #数据增强方法
        self.prepare = ConvertCocoPolysToMask(return_masks) #仅在return_masks为True时将多边形转换为掩码
        #这个对象是将数据标注的多边形(polygon)坐标转换为掩码，但其实不仅仅是这样，或者说不一定是这样，因为需要根据传进去的参数 return_masks 来确定

    #由于coco数据集中的标注格式为：
    ''' annotation{"id": int, 
    "image_id": int, 
    "category_id": int,
    "segmentation": RLE or [polygon],
    "area": float,
    "bbox": [x,y,width,height], 
    "iscrowd": 0 or 1}
    当 "iscrowd" 字段为0时，segmentation就是polygon的形式，比如这时的 "segmentation" 的值可能为 [[510.66, 423.01, 511.72, 420.03, 510.45......], ..]，
    其中是一个个polygon即多边形，这些数按序两两组成多边形各个点的横、纵坐标，也就是说，表示polygon的list中如果有n个数（必定是偶数），
    那么就代表了 n/2 个点坐标
    '''

    def __getitem__(self, idx):
        img, target = super(CocoDetection, self).__getitem__(idx)
        image_id = self.ids[idx]
        target = {'image_id': image_id, 'annotations': target}
        img, target = self.prepare(img, target)
        if self._transforms is not None:
            img, target = self._transforms(img, target)
        return img, target


def convert_coco_poly_to_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width) #将每个多边形结合图像尺寸解码为掩码
        mask = coco_mask.decode(rles) #然后将掩码增加至3维(如果之前不足3维的话)
        #为何要加一维呢？因为我们希望的是这个mask能够在图像尺寸范围（h, w）中指示每个点为0或1，
        # 在解码后，mask的shape应该是 (h,w)，加一维变为 (h,w,1)，然后在最后一个维度使用any()后才能维持原来的维度即(h,w)；
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        #将一个个多边形转换得到的掩码添加至列表，堆叠起来形成张量后返回
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks


class ConvertCocoPolysToMask(object):
    def __init__(self, return_masks=False):
        self.return_masks = return_masks

    def __call__(self, image, target):
        # target 是一个list，其中包含了多个字典类型的annotation，每个annotation的格式如上所示
        w, h = image.size

        image_id = target["image_id"]
        image_id = torch.tensor([image_id])

        anno = target["annotations"]

        anno = [obj for obj in anno if 'iscrowd' not in obj or obj['iscrowd'] == 0] #过滤掉"iscrowd"为1的数据，仅保留标注为单个对象的数据

        boxes = [obj["bbox"] for obj in anno]
        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        #将bbox的左下角和右下角坐标控制在图像尺寸范围内
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        classes = [obj["category_id"] for obj in anno]
        classes = torch.tensor(classes, dtype=torch.int64)

        if self.return_masks:
            segmentations = [obj["segmentation"] for obj in anno]
            #根据标注的segmentaion字段生成掩码，用于分割任务.
            masks = convert_coco_poly_to_mask(segmentations, h, w)

        keypoints = None
        if anno and "keypoints" in anno[0]:
            keypoints = [obj["keypoints"] for obj in anno]
            keypoints = torch.as_tensor(keypoints, dtype=torch.float32)
            num_keypoints = keypoints.shape[0]
            if num_keypoints:
                keypoints = keypoints.view(num_keypoints, -1, 3)

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]
        if self.return_masks: #keep代表那些有效的bbox，即左上角坐标小于右下角坐标那些，过滤掉无效的那批。
            masks = masks[keep]
        if keypoints is not None:
            keypoints = keypoints[keep]

        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        if self.return_masks:
            target["masks"] = masks
        target["image_id"] = image_id
        if keypoints is not None:
            target["keypoints"] = keypoints

        # for conversion to coco api
        area = torch.tensor([obj["area"] for obj in anno])
        iscrowd = torch.tensor([obj["iscrowd"] if "iscrowd" in obj else 0 for obj in anno])
        target["area"] = area[keep]
        target["iscrowd"] = iscrowd[keep]

        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        target["size"] = torch.as_tensor([int(h), int(w)])

        return image, target


def make_coco_transforms(image_set):

    normalize = T.Compose([
        T.ToTensor(), #将图像的通道维度排列在第一个维度，并且像素值归一化到0-1范围内
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) #根据指定的均值和标准差对图像进行归一化，同时将标签的bbox转换为 [公式] 形式后归一化到0-1，
    ])

    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]

    if image_set == 'train':
        return T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomSelect(
                T.RandomResize(scales, max_size=1333),
                T.Compose([
                    T.RandomResize([400, 500, 600]),
                    T.RandomSizeCrop(384, 600),
                    T.RandomResize(scales, max_size=1333),
                ])
            ),
            normalize,
        ])

    if image_set == 'val':
        return T.Compose([
            T.RandomResize([800], max_size=1333),
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')


def build(image_set, args):
    root = Path(args.coco_path)
    assert root.exists(), f'provided COCO path {root} does not exist'
    #目标检测，COCO数据集的标注文件是'annotation/instances_xxx2017.json'
    mode = 'instances'
    PATHS = {
        "train": (root / "train2017", root / "annotations" / f'{mode}_train2017.json'),
        "val": (root / "val2017", root / "annotations" / f'{mode}_val2017.json'),
    }
    #image_set表征验证集还是训练集
    img_folder, ann_file = PATHS[image_set]
    #使用COCO　API来构建数据集
    dataset = CocoDetection(img_folder, ann_file, transforms=make_coco_transforms(image_set), return_masks=args.masks)
    return dataset
