""" 
@Date: 2022/07/21
@description:
"""
import torch
import torch.nn as nn
from torch.nn import functional as F


class SegLoss(nn.Module):
    def __init__(self, label_weight=[0.8, 1], reduction='mean'):
        super(SegLoss, self).__init__()
        self.reduction = reduction

        self.label_weight = torch.tensor(label_weight).cuda()
        self.register_buffer('weight_seg', self.label_weight[1] / self.label_weight[0])

    def forward(self, gt, dt):
        gt_seg = gt['segmentation']
        dt_seg = dt['segmentation']

        # gt_seg = gt_seg.unsqueeze(1)
        dt_seg = F.interpolate(dt_seg, size=gt_seg.shape[-2:], mode='bilinear', align_corners=False)
        # gt_seg = F.interpolate(gt_seg, size=dt_seg.shape[-2:], mode='bilinear', align_corners=True)
        # gt_seg = gt_seg.squeeze(1)
        dt_seg = dt_seg.squeeze(1)

        gt_seg = gt_seg.float()

        loss = F.binary_cross_entropy_with_logits(dt_seg, gt_seg, pos_weight=self.weight_seg,
                                                  reduction='none') / self.weight_seg
        loss = loss.mean()

        return loss


if __name__ == '__main__':
    input = torch.randn(4, 1, 8, 16).cuda()
    target = torch.zeros(4, 8, 16).cuda()

    a = SegLoss()

    loss = a(target, input)
    print(loss)
