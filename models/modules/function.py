""" 
@Date: 2021/09/01
@description:
"""
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms

from matplotlib import pyplot as plt
import functools


def lr_pad(x, padding=1):
    ''' Pad left/right-most to each other instead of zero padding '''
    return torch.cat([x[..., -padding:], x, x[..., :padding]], dim=3)


class LR_PAD(nn.Module):
    ''' Pad left/right-most to each other instead of zero padding '''

    def __init__(self, padding=1):
        super(LR_PAD, self).__init__()
        self.padding = padding

    def forward(self, x):
        return lr_pad(x, self.padding)


def wrap_lr_pad(net):
    for name, m in net.named_modules():
        if not isinstance(m, nn.Conv2d):
            continue
        if m.padding[1] == 0:
            continue
        w_pad = int(m.padding[1])
        m.padding = (m.padding[0], 0)  # weight padding is 0, LR_PAD then use valid padding will keep dim of weight
        names = name.split('.')

        root = functools.reduce(lambda o, i: getattr(o, i), [net] + names[:-1])
        setattr(
            root, names[-1],
            nn.Sequential(LR_PAD(w_pad), m)
        )


def pano_upsample_w(x, s):
    if len(x.shape) == 3:
        mode = 'linear'
        scale_factor = s
    elif len(x.shape) == 4:
        mode = 'bilinear'
        scale_factor = (1, s)
    else:
        raise NotImplementedError
    x = torch.cat([x[..., -1:], x, x[..., :1]], dim=-1)
    x = F.interpolate(x, scale_factor=scale_factor, mode=mode, align_corners=False)
    x = x[..., s:-s]
    return x


class PanoUpsampleW(nn.Module):
    def __init__(self, s):
        super(PanoUpsampleW, self).__init__()
        self.s = s

    def forward(self, x):
        return pano_upsample_w(x, self.s)


def weight_init(module):
    for n, m in module.named_children():
        print('initialize: '+n)
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.ones_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Sequential):
            for f, g in m.named_children():
                print('initialize: ' + f)
                if isinstance(g, nn.Conv2d):
                    nn.init.kaiming_normal_(g.weight, mode='fan_in', nonlinearity='relu')
                    if g.bias is not None:
                        nn.init.zeros_(g.bias)
                elif isinstance(g, (nn.BatchNorm2d, nn.GroupNorm)):
                    nn.init.ones_(g.weight)
                    if g.bias is not None:
                        nn.init.zeros_(g.bias)
                elif isinstance(g, nn.Linear):
                    nn.init.kaiming_normal_(g.weight, mode='fan_in', nonlinearity='relu')
                    if g.bias is not None:
                        nn.init.zeros_(g.bias)
        elif isinstance(m, nn.AdaptiveAvgPool2d) or isinstance(m, nn.AdaptiveMaxPool2d) or isinstance(m, nn.ModuleList) or isinstance(m, nn.BCELoss):
            a=1
        else:
            m.initialize()


def show_feature_map(feature_map, mode='mean'):
    if mode == 'mean':
        feature_map = feature_map.mean(1).squeeze(0).cpu()
    else:
        feature_map = torch.max(feature_map, dim=1)[1].squeeze(0).cpu().float()

    plt.figure()
    plt.imshow(transforms.ToPILImage()(feature_map))
    plt.axis('off')

    plt.show()
