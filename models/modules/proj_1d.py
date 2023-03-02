""" 
@author:
@Date: 2022/06/27
@description: Projection Based on 1D Features
"""

import torch
import torch.nn as nn
from torch.nn import functional as F

from models.modules.function import *


class Projection1D(torch.nn.Module):
    def __init__(self, num_layers, in_channels, bias=True, k=3):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_layers = nn.ModuleList(nn.Conv1d(in_channels if i == 0 else in_channels*2,
                                                     in_channels*2 if i == 0 else in_channels*4,
                                                     kernel_size=k, bias=bias, padding=(k - 1) // 2)
                                           for i in range(num_layers))
        self.hidden_norms = nn.ModuleList(nn.BatchNorm1d(in_channels * 2 ** (i + 1)) for i in range(num_layers))

        self.up_1d = PanoUpsampleW(2)

    def forward(self, x):
        for conv, norm in zip(self.hidden_layers, self.hidden_norms):
            x = F.relu(norm(conv(self.up_1d(x))))
            # x = F.relu(norm(conv(x)))

        return x
