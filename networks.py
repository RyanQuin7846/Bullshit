# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 16:36:25 2024

@author: Server2
"""

import torch
import torch.nn as nn
import torch.nn.init as init

class FCNN(nn.Module):
    def __init__(self):
        super(FCNN, self).__init__()
        self.act2 = nn.Softplus()
        self.linear1 = nn.Linear(5, 256)
        self.linear2 = nn.Linear(256, 256)
        self.linear3 = nn.Linear(256, 256)
        self.linear4 = nn.Linear(256, 256)
        self.linear5 = nn.Linear(256, 256)
        self.linear6 = nn.Linear(256, 1)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.uniform_(m.weight, a=-2, b=2)
                init.constant_(m.bias, 0)
    def forward(self, x,y,z,a,b):
        z = torch.cat((x,y,z,a,b), axis=1)
        # z = x
        out = torch.sigmoid(self.linear1(z))
        out = torch.sigmoid(self.linear2(out))
        out = torch.sigmoid(self.linear3(out))
        out = torch.sigmoid(self.linear4(out))
        out = torch.sigmoid(self.linear5(out))
        out = 2.6 * torch.sigmoid(self.linear6(out)) + 0.1
        # out = 3*torch.sigmoid(self.linear6(out))
        return out