# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 16:00:31 2024

@author: Server2
"""

import torch
import torch.nn as nn
import torch.nn.init as init

class FCNN(nn.Module):
    def __init__(self):
        super(FCNN, self).__init__()
        self.act2 = nn.Softplus()

        # 定义第一层和最后一层
        self.linear1 = nn.Linear(2, 256)
        self.linear_last = nn.Linear(256, 1)

        # 定义28个中间层，每层输入输出维度均为256
        self.middle_layers = nn.ModuleList([nn.Linear(256, 256) for _ in range(28)])

        # 初始化参数
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.uniform_(m.weight, a=-2, b=2)
                init.constant_(m.bias, 0)

    def forward(self, x, y):
        z = torch.cat((x, y), axis=1)

        # 通过第一层
        out = torch.sigmoid(self.linear1(z))

        # 通过28个中间层
        for layer in self.middle_layers:
            out = torch.sigmoid(layer(out))

        # 通过最后一层
        out = self.linear_last(out)

        return out