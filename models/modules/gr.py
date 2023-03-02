""" 
@author: Zishuo zheng
@Date: 2022/09/09
@description: Attention Graph Reasoning
"""

import math
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.nn.functional as F

from torch import nn, einsum
from einops import rearrange

# from models.modules.utils import *


class AttnGlobalReasoning(nn.Module):
    def __init__(self, dim, depth, mlp_dim, dropout=0.):
        super().__init__()

        self.layers = nn.ModuleList([])
        # self.layers_self = nn.ModuleList([])
        # self.layers_cross = nn.ModuleList([])

        self.gr_d = GR(dim)
        self.gr_r = GR(dim)


        # self.layers_self_d = nn.ModuleList([
        #     Attention(dim) for _ in range(depth)
        # ])
        # self.layers_self_r = nn.ModuleList([
        #     Attention(dim) for _ in range(depth)
        # ])

        # self.layers_cros_d = nn.ModuleList([
        #     Attention(dim) for _ in range(depth)
        # ])
        # self.layers_cros_r = nn.ModuleList([
        #     Attention(dim) for _ in range(depth)
        # ])

        for i in range(depth):
            self.layers.append(nn.ModuleList([
                PreNormAtt(dim, Attention(dim)),
                PreNormAtt(dim, Attention(dim)),
                PreNormAtt(dim, Attention(dim)),
                PreNormAtt(dim, Attention(dim)),
                PreNormFF(dim, FeedForward(dim, mlp_dim, dropout=dropout)),
                PreNormFF(dim, FeedForward(dim, mlp_dim, dropout=dropout)),
                PreNormFF(dim, FeedForward(dim, mlp_dim, dropout=dropout)),
                PreNormFF(dim, FeedForward(dim, mlp_dim, dropout=dropout)),
            ]))

        # self.ff = PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))

    def forward(self, x_d, x_r):
        x_d = self.gr_d(x_d) + x_d
        x_r = self.gr_r(x_r) + x_r

        # for sa, ca in zip(self.layers_self, self.layers_cros):
        #     d_self, r_self = sa(x_d, x_d), sa(x_r, x_r)
        #     x_d, x_r = (x_d + d_self), (x_r + r_self)

        #     d_cros, r_cros = ca(x_d, x_r), ca(x_r, x_d)
        #     x_d, x_r = (x_d + d_cros), (x_r + r_cros)

        for sa_d, sa_r, ca_d, ca_r, ff_d_s, ff_d_c, ff_r_s, ff_r_c in self.layers:
            # self attention
            d_self, r_self = sa_d(x_d, x_d), sa_r(x_r, x_r)
            x_d, x_r = (x_d + d_self), (x_r + r_self)

            x_d = ff_d_s(x_d) + x_d
            x_r = ff_r_s(x_r) + x_r

            # cross attention
            d_cros, r_cros = ca_d(x_d, x_r), ca_r(x_r, x_d)
            x_d, x_r = (x_d + d_cros), (x_r + r_cros)

            x_d = ff_d_c(x_d) + x_d
            x_r = ff_r_c(x_r) + x_r

        return x_d, x_r


class GR(nn.Module):
    def __init__(self, in_channels, squeeze_ratio=4):
        super(GR, self).__init__()
        self.fc = nn.Linear(in_channels, in_channels // squeeze_ratio)
        self.ca_diag = ChannelAttention_diag(in_channels, squeeze_ratio=squeeze_ratio)

        self.GCN = GraphConvolution(in_channels, in_channels)

    def forward(self, x):
        device = torch.device("cuda")

        B, W, C = x.shape
        b = torch.unsqueeze(torch.eye(W, device=device), 0).expand(B, W, W).cuda()
        One = torch.ones(W, 1, dtype=torch.float32, device=device).expand(B, W, 1).cuda()
        diag = self.ca_diag(x.permute(0, 2, 1).unsqueeze(2))
        xx = self.fc(x)

        x_k = xx.permute(0, 2, 1)
        x_q = xx

        D = torch.bmm(x_q, diag)
        D = torch.sigmoid(torch.bmm(D, x_k))
        D = torch.bmm(D, One)
        D = D ** (-1 / 2)
        D = torch.mul(b, D)

        P = torch.bmm(D, x_q)
        Pt = P.permute(0, 2, 1)

        X = x
        LX = torch.bmm(Pt, X)
        LX = torch.bmm(diag, LX)
        LX = torch.bmm(P, LX)
        LX = X - LX

        Y = self.GCN(LX)

        return Y

    def initialize(self):
        weight_init(self)


class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight = Parameter(torch.FloatTensor(in_features, out_features))

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, input):
        output = torch.matmul(input, self.weight)

        return output


class ChannelAttention_diag(nn.Module):
    def __init__(self, in_dim, squeeze_ratio=4):
        super(ChannelAttention_diag, self).__init__()
        self.inter_dim = in_dim // squeeze_ratio
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.avg_fc = nn.Sequential(nn.Linear(in_dim, self.inter_dim, bias=False),
                                    nn.ReLU(inplace=True)
                                    )
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.max_fc = nn.Sequential(nn.Linear(in_dim, self.inter_dim, bias=False),
                                    nn.ReLU(inplace=True)
                                    )

    def forward(self, x):
        device = torch.device("cuda")
        B, C, H, W = x.size()
        M = self.inter_dim
        x_avg = self.avg_fc(self.avg_pool(x).squeeze())
        x_max = self.max_fc(self.max_pool(x).squeeze())
        cw = torch.sigmoid(x_avg + x_max).unsqueeze(-1)
        b = torch.unsqueeze(torch.eye(M, device=device), 0).expand(B, M, M).cuda()
        return torch.mul(b, cw)

    def initialize(self):
        weight_init(self)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class PreNormFF(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x):
        return self.fn(self.norm(x))

class PreNormAtt(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x1, x2):
        return self.fn(self.norm1(x1), self.norm2(x2))


class GELU(nn.Module):
    def forward(self, input):
        return F.gelu(input)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dropout=0.):
        super().__init__()

        dim_head = dim // heads

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, source):
        b, n, _, h = *x.shape, self.heads
        q, k, v = self.to_q(x), self.to_k(source), self.to_v(source)
        q = rearrange(q, 'b n (h d) -> b h n d', h=h)
        k = rearrange(k, 'b n (h d) -> b h n d', h=h)
        v = rearrange(v, 'b n (h d) -> b h n d', h=h)

        scores = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = F.softmax(scores, dim=-1, dtype=scores.dtype)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
    
        return self.to_out(out)
