# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR Transformer class.

Copy-paste from torch.nn.Transformer with modifications:
    * positional encodings are passed in MHattention
    * extra LN at the end of encoder is removed
    * decoder returns a stack of activations from all decoding layers
"""
import copy
from typing import Optional, List

import torch
import torch.nn.functional as F
from torch import nn, Tensor


class Transformer(nn.Module):

    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False):
        super().__init__()
        #构建Encoder
        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        # 如果是后归一化的方式，那么Encoder每层输出都会进行归一化，
        # 因此在Encoder对最后的输出就不需要再额外进行归一化了，这种情况下就将encoder_norm设为None
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)
        #构建Decoder
        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        #　在Decoder中decoder_norm始终存在
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm,
                                          return_intermediate=return_intermediate_dec)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, mask, query_embed, pos_embed):
        '''tgt是与query embedding形状一直且设置为全0的结果，意为初始化需要预测的目标。因为一开始并不清楚这些目标，
        所以初始化为全0。其会在Decoder的各层不断被refine，相当于一个coarse-to-fine的过程，但是真正要学习的是query embedding，
        学习到的是整个数据集中目标物体的统计特征，而tgt在每次迭代训练（一个batch数据刚到来）时会被重新初始化为0'''
        # flatten NxCxHxW to HWxNxC
        #　这里c与hidden_dim相等
        bs, c, h, w = src.shape
        # (h*w, bs, c=hidden_dim)
        src = src.flatten(2).permute(2, 0, 1)
        # (h*w, bs, c=hidden_dim)
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        # (num_queries, bs, hidden_dim)
        query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)
        # (bs, h*w)
        mask = mask.flatten(1)
        # (nun_queries, bs, hidden_dim)
        tgt = torch.zeros_like(query_embed) #将query embedding与网络输出的目标尺寸保持一致
        # (h*w, bs, c=hidden_dim)
        memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)
        # 在项目整体代码中，TransformerDecoder的初始化参数return_intermediate设置为True,
        # 因此Decoder的输出包含了每层的结果，shape是(6, num_queries, bs, hidden_dim)
        hs = self.decoder(tgt, memory, memory_key_padding_mask=mask,
                          pos=pos_embed, query_pos=query_embed)
        #(6, bs, num_queries, hidden_dim) (bs, c=hidden_dim, h, w)
        return hs.transpose(1, 2), memory.permute(1, 2, 0).view(bs, c, h, w)


class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        #由于Encoder通常有六层，每层结构相同，这里使用get_clones方法复制
        self.layers = _get_clones(encoder_layer, num_layers) #_get_clones() 方法将结构相同的层复制（注意是deepcopy）多次，返回一个nn.ModuleList实例
        self.num_layers = num_layers
        self.norm = norm #归一化层

    def forward(self, src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        #src对应backbone最后一层输出的特征图，并且维度映射到了hidden_dim, shape是(h*w,b,hidden_dim)
        #pos对应backbone最后一层输出的特征图对应的位置编码, shape是(h*w,b,c)
        #src_key_padding_mask对应backbone最后一层输出的特征图对应的mask, shape是(b, h*w)
        output = src

        for layer in self.layers: #循环调用每层的前向过程，前一次层的输出作为后一层的输入
            output = layer(output, src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask, pos=pos)

        if self.norm is not None: #需要归一化的话对最后一层输出做归一化
            output = self.norm(output)

        return output


class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        #是否需要记录中间每层的结果
        self.return_intermediate = return_intermediate

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        output = tgt
        # tgt是query embedding, shape是(num_queires, b, hidden_dim)
        # query_pos是对应tgt的位置编码，shape和tgt一致
        # memeory是Encoder的输出，shape是(h*w, b, hidden_dim)
        # memeory_key_padding_mask对应Encoder的src_key_padding_mask，也是EncoderLayer的key_padding_mask，shape是(b, h*w)
        # pos对应输入到Encoder的位置编码，这里代表memory的位置编码，shape与memeory一致
        intermediate = []

        for layer in self.layers:
            output = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos)
            if self.return_intermediate:
                #intermediate中记录的是每层输出后的归一化结果，而每一层的输入是前一层输出（没有归一化）的结果
                intermediate.append(self.norm(output))

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output.unsqueeze(0)


class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        #多头自注意力层
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        #　FFN前向反馈层
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        #分别用于多头自注意力层和前向反馈层
        self.norm1 = nn.LayerNorm(d_model) #层归一化
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation) #_get_activation_fn() 方法根据输入参数指定的激活方式返回对应的激活层，默认是ReLU
        #是否在输入多头自注意力层/前向反馈层前进行归一化
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        #在输入多头自注意力层时需要先进行位置嵌入，即结合位置编码。注意仅对query和key实施，而value不需要。
        # query和key是在图像特征中各个位置之间计算相关性，而value作为原图像特征，使用计算出来的相关性加权上去，
        # 得到各位置结合了全局相关性（增强/削弱）后的特征表示。
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     src,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None):
        #在输入多头注意力层和前向反馈层输出后归一化
        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def forward_pre(self, src,
                    src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(q, k, value=src2, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        #前向过程输出两部分，一个是自注意力层的输出，一个是自注意力权重　[0] 表示提取的为自注意力层的输出
        #此处的src_key_padding_mask是backbone最后一层输出特征图对应的mask.
        # 值为True的那些位置代表原始图像padding的部分，在生成注意力的过程中会被填充为-inf,
        # 这样最终生成注意力经过softmax时输出就趋向于0，相当于忽略不计
        # src_mask 是在Transformer中用来“防作弊”的，即遮住当前预测位置之后的位置，忽略这些位置，不计算与其相关的注意力权重
        #而这里的序列是图像特征（反而就是要计算图像各位置的全局关系）

        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(self, src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        if self.normalize_before: #根据是否在输入多头注意力层和前向反馈层前进行归一化来选择不同的前向传播方式
            #一种是在输入多头自注意力层和前向反馈层前先进行归一化，另一种则是在这两个层输出后再进行归一化操作
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)


class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos #直接将query_embedding的权重与特征相加

    def forward_post(self, tgt, memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        # 首先进行位置嵌入
        q = k = self.with_pos_embed(tgt, query_pos)
        # 多头自注意力层，输入参数不包含Encoder的输出
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        #　该层后在进行归一化
        tgt = self.norm1(tgt)
        # Encoder-Deconder层，key和value来自Encoder的输出， query来自上一层的输出，query和key需要进行位置嵌入，
        # memory_key_padding_mask对应EncoderLayer的key_padding_mask
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        #该层输出后再归一化
        tgt = self.norm2(tgt)
        # FFN
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        ##该层输出后再归一化
        tgt = self.norm3(tgt)
        "这里自注意力层计算的相关性是目标物体与图像特征各位置的相关性，" \
        "然后再把这个相关性系数加权到Encoder编码后的图像特征（value）上，" \
        "相当于获得了object features的意思，更好地表征了图像中的各个物体"
        return tgt

    def forward_pre(self, tgt, memory,
                    tgt_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def build_transformer(args):
    return Transformer(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        normalize_before=args.pre_norm,
        return_intermediate_dec=True,
    )


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")
