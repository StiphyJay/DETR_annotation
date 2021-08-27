# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Various positional encodings for the transformer.
"""
import math
import torch
from torch import nn

from util.misc import NestedTensor


class PositionEmbeddingSine(nn.Module):
    """
    将每个位置的各个维度映射到角度上
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """
    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, tensor_list: NestedTensor):
        #[b, c, h, w]
        x = tensor_list.tensors
        #[b, h, w]
        mask = tensor_list.mask #mask 指示了图像哪些位置是padding而来的，其值为True的部分就是padding的部分
        assert mask is not None
        #图像中不是padding的部分
        not_mask = ~mask #取反后得到not_mask，那么值为True的部分就是图像真实有效（而非padding）的部分
        y_embed = not_mask.cumsum(1, dtype=torch.float32) #在第一维(行方向)累加
        x_embed = not_mask.cumsum(2, dtype=torch.float32) #在第二维(列方向)累加
        if self.normalize:
            eps = 1e-6
            #列方向上做归一化
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            #行方向上做归一化
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        #pow(10000, 2i/d)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)
        #[b, c, h, w, num_pos_feats]
        pos_x = x_embed[:, :, :, None] / dim_t
        #[b, c, h, w, num_pos_feats]
        pos_y = y_embed[:, :, :, None] / dim_t
        #最后一维中，偶数维上使用正弦编码，奇数维上使用余弦编码
        #[b, c, h, w, num_pos_feats // 2, 2] -> [b, h, w, 2*(num_pos_feats // 2)]
        #使用这种方式编码，即将各行各列的奇偶维度分别进行正,余弦编码
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        #对于每个位置(x,y)，其所在列对应的编码值排在通道这个维度上的前 num_pos_feats 维，
        # 而其所在行对应的编码值则排在通道这个维度上的后 num_pos_feats 维。
        # 这样，特征图上的各位置（总共 h*w 个）都对应到不同的维度为 2*num_pos_feats 的编码值（向量）。
        return pos


class PositionEmbeddingLearned(nn.Module):
    """
    Absolute pos embedding, learned.
    """
    def __init__(self, num_pos_feats=256):
        super().__init__()
        #默认需要编码的行列位置不超过50个 即位置索引在0~50范围内，对每个位置都嵌入到 num_pos_feats（默认256）维。
        self.row_embed = nn.Embedding(50, num_pos_feats)
        self.col_embed = nn.Embedding(50, num_pos_feats)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.row_embed.weight)
        nn.init.uniform_(self.col_embed.weight)

    def forward(self, tensor_list: NestedTensor):
        x = tensor_list.tensors
        h, w = x.shape[-2:]
        #一行中的每个位置
        i = torch.arange(w, device=x.device)
        #一列中的每个位置
        j = torch.arange(h, device=x.device)
        x_emb = self.col_embed(i)
        y_emb = self.row_embed(j)
        #最后将行、列编码结果拼接起来并扩充第一维，与batch size对应，得到以下变量 pos (N,num_pos_feats*2, h, w)
        # 所有行同一列的横坐标（x_emb）编码结果是一样的，在dim1中处于 pos 的前 num_pos_feats 维；
        # 同理，所有列所有列同一行的纵坐标（y_emb）编码结果也是一样的，在dim1中处于 pos 的后 num_pos_feats 维
        pos = torch.cat([
            x_emb.unsqueeze(0).repeat(h, 1, 1),
            y_emb.unsqueeze(1).repeat(1, w, 1),
        ], dim=-1).permute(2, 0, 1).unsqueeze(0).repeat(x.shape[0], 1, 1, 1)
        return pos


def build_position_encoding(args):
    N_steps = args.hidden_dim // 2
    if args.position_embedding in ('v2', 'sine'):
        #余弦编码
        # TODO find a better way of exposing other arguments
        position_embedding = PositionEmbeddingSine(N_steps, normalize=True)
    elif args.position_embedding in ('v3', 'learned'):
        #可学习的绝对编码方式
        position_embedding = PositionEmbeddingLearned(N_steps)
    else:
        raise ValueError(f"not supported {args.position_embedding}")

    return position_embedding
