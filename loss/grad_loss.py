""" 
@Date: 2021/08/12
@description:
"""
import torch
import torch.nn as nn
import numpy as np

from visualization.grad import get_all


class GradLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.L1Loss()
        self.cos = nn.CosineSimilarity(dim=-1, eps=0)

        self.grad_conv = nn.Conv1d(1, 1, kernel_size=3, stride=1, padding=0, bias=False, padding_mode='circular')
        self.grad_conv.weight = nn.Parameter(torch.tensor([[[1, 0, -1]]]).float())
        self.grad_conv.weight.requires_grad = False

    def forward(self, gt, dt):
        gt_direction, _, gt_angle_grad = get_all(gt['depth'], self.grad_conv)
        dt_direction, _, dt_angle_grad = get_all(dt['depth'], self.grad_conv)

        normal_loss = (1 - self.cos(gt_direction, dt_direction)).mean()
        grad_loss = self.loss(gt_angle_grad, dt_angle_grad)
        return [normal_loss, grad_loss]
