import torch.nn
import torch
import torch.nn as nn
import models.modules as modules
import numpy as np

from models.base_model import BaseModule
from models.modules.my_feature_extractor import MyFeatureExtractor
from models.modules.flip_fusion import FeatureFlipFusion
from models.modules.seg_head import SegHead
from models.modules.proj_1d import Projection1D
from models.modules.swg_transformer import SWG_Transformer
from models.modules.gr import AttnGlobalReasoning
from models.modules.function import *

from utils.conversion import uv2depth, get_u, lonlat2depth, get_lon, lonlat2uv
from utils.height import calc_ceil_ratio
from utils.misc import tensor2np
from PSAutils.equisamplingpoint import genSamplingPattern
from PSAutils.DBAT import *
from einops.layers.torch import Rearrange
from torch import einsum

class FastLeFF(nn.Module):
    
    def __init__(self, dim=256, hidden_dim=512, act_layer=nn.GELU,drop = 0.):
        super().__init__()

        #from torch_dwconv import depthwise_conv2d, DepthwiseConv2d

        self.linear1 = nn.Sequential(nn.Linear(dim, hidden_dim),
                                act_layer())
        self.dwconv = nn.Sequential(nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3,stride=1,padding=1),
                        act_layer())
        self.linear2 = nn.Sequential(nn.Linear(hidden_dim, dim))
        self.dim = dim
        self.hidden_dim = hidden_dim

    def forward(self, x, hh, ww):
        # bs x hw x c
        bs, hw, c = x.size()

        x = self.linear1(x)

        # spatial restore
        x = x.view(bs, hh, ww, self.hidden_dim)
        x = x.permute(0, 3, 1, 2)
        # bs,hidden_dim,32x32

        x = self.dwconv(x)

        # flaten
        x = x.view(bs, self.hidden_dim, hh*ww)
        x = x.permute(0, 2, 1)#Rearrange(x, ' b c h w -> b (h w) c', h = hh, w = ww)

        x = self.linear2(x)

        return x

class My_Layout_Net(BaseModule):
    def __init__(self, ckpt_dir=None, backbone='resnet34', dropout=0.0, corner_heat_map=False):
        super().__init__(ckpt_dir)

        self.patch_num = 256
        self.patch_dim = 1024
        
        self.ref_point16x32 = genSamplingPattern(32, 64, 3, 3).cuda()
        self.norm1 = nn.LayerNorm(256)
        self.norm2 = nn.LayerNorm(256)
        self.norm3 = nn.LayerNorm(256)
        self.norm4 = nn.LayerNorm(256)
        self.dattn1 = DeformableHeadAttention(8, 256, k=9, last_feat_height=32, last_feat_width=64, scales=4, dropout=0.0, need_attn=False)
        self.dattn2 = DeformableHeadAttention(8, 256, k=9, last_feat_height=32, last_feat_width=64, scales=1, dropout=0.0, need_attn=False)
        self.mlp1 = FastLeFF()
        self.mlp2 = FastLeFF()

        self.corner_heat_map = corner_heat_map
        self.dropout_d = dropout

        self.feature_extractor = MyFeatureExtractor(backbone)
        self.seg_head = SegHead(256, 64, 1)
        self.fusion = FeatureFlipFusion(channels=256)
        self.height_avgpool = nn.AvgPool2d(kernel_size=(32, 1), stride=1, padding=0)
        self.height_maxpool = nn.MaxPool2d(kernel_size=(32, 1), stride=1, padding=0)

        self.channel_reduce = nn.Sequential(nn.Conv2d(1024, 256, 1),
                                            nn.ReLU(inplace=True),
                                            nn.BatchNorm2d(256))

        # self.width_pool = nn.AvgPool1d(kernel_size=256, stride=1, padding=0)
        self.projection_head_d = Projection1D(2, 256)
        self.projection_head_r = Projection1D(2, 256)

        # self.process = SWG_Transformer(dim=256, depth=8, heads=8, dim_head=32, mlp_dim=1024,
        #                                win_size=16, patch_num=256, rpe='lr_parameter_mirror')

        self.gnn = AttnGlobalReasoning(dim=1024, depth=3, mlp_dim=2048)
        # self.dr_process = GlobalReasoning(dim=256, depth=4, mlp_dim=1024, mode='depth')
        # self.rr_process = GlobalReasoning(dim=256, depth=4, mlp_dim=1024, mode='ratio')

        self.proj_depth = nn.Linear(1024, 1)
        self.proj_ratio_dim = nn.Linear(1024, 1)
        self.proj_ratio = nn.Linear(256, 1)
        # // self.proj_depth = nn.Conv1d(256, 1, kernel_size=1, bias=True, padding=0)
        # // self.proj_ratio = nn.Linear(in_features=256, out_features=1)

        wrap_lr_pad(self)

        self.name = "My_Layout_Net"

    def forward(self, x):
        b,_,_,_ = x.shape
        x = self.feature_extractor(x)
        q = x[-2]
        _, h, w, c = q.shape
        #res1 = q.view(b, c, h, w)
        res1 = q.permute(0, 3, 1, 2).contiguous()
        
        q = q.view(b, h*w, c)
        tmp = q
        q = self.norm1(q) 
        q = q.view(b, h, w, c)
        x = self.dattn1(q, x, self.ref_point16x32.repeat(b, 1, 1, 1, 1))
        x = x.view(b, h*w, c)
        x = x + self.mlp1(self.norm2(x), h, w)
        
        #q = x.view(b, h*w, c)
        q = self.norm3(x + tmp) 
        q = q.view(b, h, w, c)
        x = self.dattn2(q, q.unsqueeze(0), self.ref_point16x32.repeat(b, 1, 1, 1, 1))
        x = x.view(b, h*w, c)
        x = x + self.mlp2(self.norm4(x), h, w)
        
        
        #x = x.view(b, h, w, c).permute(0, 3, 1, 2)
        x = x.permute(0, 2, 1).contiguous()
        x = x.view(b, c, h, w)
        x = x + res1
        tmp2 = x
        #x = self.channel_reduce(x)
        x = self.fusion(x.float())

        seg = self.seg_head(x)
        seg_sig = torch.sigmoid(seg)
        mask = seg_sig >= 0.5
        x_d = x * mask + tmp2
        x_r = x * ~mask + tmp2

        x_d_avg = self.height_avgpool(x_d)[:, :, 0, :]
        x_d_max = self.height_maxpool(x_d)[:, :, 0, :]
        x_r_avg = self.height_avgpool(x_r)[:, :, 0, :]
        x_r_max = self.height_maxpool(x_r)[:, :, 0, :]

        x_d = x_d_avg + x_d_max
        x_r = x_r_avg + x_r_max
        # x_r = self.width_pool(self.height_pool(x_r)[:, :, 0, :])[:, :, 0]

        x_d = self.projection_head_d(x_d)
        x_r = self.projection_head_r(x_r)

        x_d = x_d.permute(0, 2, 1).contiguous()
        x_r = x_r.permute(0, 2, 1).contiguous()

        x_d, x_r = self.gnn(x_d, x_r)

        depth = self.proj_depth(x_d)
        depth = depth.view(-1, 256)

        ratio = self.proj_ratio_dim(x_r).view(-1, 256)
        ratio = self.proj_ratio(ratio)

        output = {
            'segmentation': seg,
            'depth': depth,
            'ratio': ratio
        }

        return output
