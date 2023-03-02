""" 
@author:
@Date: 2022/05/23
@description: Feature extractor
"""
#11
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from models.modules.function import *


class Resnet(nn.Module):
    def __init__(self, backbone='resnet50', pretrained=True):
        super(Resnet, self).__init__()
        self.encoder = getattr(models, backbone[-8:])(pretrained=pretrained)
        self.horizon_1 = nn.Sequential(
            nn.Conv2d(64, 256, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.horizon_2 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.horizon_3 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.horizon_4 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        del self.encoder.fc, self.encoder.avgpool

    def forward(self, x):
        features = []
        x = self.encoder.conv1(x)
        x = self.encoder.bn1(x)
        x = self.encoder.relu(x)
        x = self.encoder.maxpool(x)

        x = self.encoder.layer1(x)
        x1 = self.horizon_1(x)
        b, c, h, w = x1.shape
        #print(x.shape)
        features.append(x1.permute(0, 2, 3, 1).contiguous())  # 1/4
        x = self.encoder.layer2(x)
        x2 = self.horizon_2(x)
        b, c, h, w = x2.shape
        #print(x.shape)
        features.append(x2.permute(0, 2, 3, 1).contiguous())  # 1/8
        x = self.encoder.layer3(x)
        x3 = self.horizon_3(x)
        b, c, h, w = x3.shape
        #print(x.shape)
        features.append(x3.permute(0, 2, 3, 1).contiguous())  # 1/16
        x = self.encoder.layer4(x)
        x4 = self.horizon_4(x)
        b, c, h, w = x4.shape
        #print(x.shape)
        features.append(x4.permute(0, 2, 3, 1).contiguous())  # 1/32

        return features


class MyFeatureExtractor(nn.Module):
    x_mean = torch.FloatTensor(np.array([0.485, 0.456, 0.406])[None, :, None, None])
    x_std = torch.FloatTensor(np.array([0.229, 0.224, 0.225])[None, :, None, None])

    def __init__(self, backbone='resnet34'):
        super(MyFeatureExtractor, self).__init__()

        self.feature_extractor = Resnet(backbone, pretrained=True)

        # def ChannelReduction(in_c, out_c):
        #     return nn.Sequential(
        #         nn.Conv2d(in_c, out_c, 1, bias=False),
        #         nn.BatchNorm2d(out_c),
        #         nn.ReLU(inplace=True)
        #     )

        # self.cr_lst = nn.ModuleList([
        #     ChannelReduction(256, 256),
        #     ChannelReduction(512, 256),
        #     ChannelReduction(1024, 256),
        #     ChannelReduction(2048, 256),
        # ])

        self.x_mean.requires_grad = False
        self.x_std.requires_grad = False

    def _prepare_x(self, x):
        x = x.clone()
        if self.x_mean.device != x.device:
            self.x_mean = self.x_mean.to(x.device)
            self.x_std = self.x_std.to(x.device)
        x[:, :3] = (x[:, :3] - self.x_mean) / self.x_std

        return x

    def forward(self, x):
        x = self._prepare_x(x)
        # x_lst = self.feature_extractor(x)
        x = self.feature_extractor(x)

        # x_lst = [f(n) for f, n in zip(self.cr_lst, x_lst)]

        # x_lst[3] = F.interpolate(x_lst[3], size=x_lst[0].size()[2:], mode='bilinear', align_corners=True)
        # x_lst[2] = F.interpolate(x_lst[2], size=x_lst[0].size()[2:], mode='bilinear', align_corners=True)
        # x_lst[1] = F.interpolate(x_lst[1], size=x_lst[0].size()[2:], mode='bilinear', align_corners=True)
        # x_lst[0] = x_lst[0]

        # x = torch.cat(x_lst, dim=1)
        # x = x_lst[0] + x_lst[1] + x_lst[2] + x_lst[3]

        return x
