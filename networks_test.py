# -*- coding: utf-8 -*-
"""
Created on Thu Aug 29 17:17:27 2024

@author: Server2
"""

import torch
import torch.nn as nn
import torch.nn.init as init

class FCNN(nn.Module):
    def __init__(self):
        super(FCNN, self).__init__()
        self.act2 = nn.ReLU()
        self.linear1 = nn.Linear(5*1296, 256)
        self.linear2 = nn.Linear(256, 256)
        self.linear3 = nn.Linear(256, 256)
        self.linear4 = nn.Linear(256, 256)
        self.linear5 = nn.Linear(256, 256)
        self.linear6 = nn.Linear(256, 256)
        self.linear7 = nn.Linear(256, 256)
        self.linear8 = nn.Linear(256, 256)
        self.linear9 = nn.Linear(256, 256)
        self.linear10 = nn.Linear(256, 256)
        self.linear11 = nn.Linear(256, 256)
        self.linear12 = nn.Linear(256, 256)
        self.linear13 = nn.Linear(256, 256)
        self.linear14 = nn.Linear(256, 256)
        self.linear15 = nn.Linear(256, 256)
        self.linear16 = nn.Linear(256, 256)
        self.linear17 = nn.Linear(256, 256)
        self.linear18 = nn.Linear(256, 256)
        self.linear19 = nn.Linear(256, 256)
        self.linear20 = nn.Linear(256, 256)
        self.linear21 = nn.Linear(256, 256)
        self.linear22 = nn.Linear(256, 256)
        self.linear23 = nn.Linear(256, 256)
        self.linear24 = nn.Linear(256, 256)
        self.linear25 = nn.Linear(256, 256)
        self.linear26 = nn.Linear(256, 256)
        self.linear27 = nn.Linear(256, 256)
        self.linear28 = nn.Linear(256, 256)
        self.linear29 = nn.Linear(256, 256)
        self.linear30 = nn.Linear(256, 1*1296)
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
        out = torch.sigmoid(self.linear6(out))
        out = torch.sigmoid(self.linear7(out))
        out = torch.sigmoid(self.linear8(out))
        out = torch.sigmoid(self.linear9(out))
        out = torch.sigmoid(self.linear10(out))
        out = torch.sigmoid(self.linear11(out))
        out = torch.sigmoid(self.linear12(out))
        out = torch.sigmoid(self.linear13(out))
        out = torch.sigmoid(self.linear14(out))
        out = torch.sigmoid(self.linear15(out))
        out = torch.sigmoid(self.linear16(out))
        out = torch.sigmoid(self.linear17(out))
        out = torch.sigmoid(self.linear18(out))
        out = torch.sigmoid(self.linear19(out))
        out = torch.sigmoid(self.linear20(out))
        out = torch.sigmoid(self.linear21(out))
        out = torch.sigmoid(self.linear22(out))
        out = torch.sigmoid(self.linear23(out))
        out = torch.sigmoid(self.linear24(out))
        out = torch.sigmoid(self.linear25(out))
        out = torch.sigmoid(self.linear26(out))
        out = torch.sigmoid(self.linear27(out))
        out = torch.sigmoid(self.linear28(out))
        out = torch.sigmoid(self.linear29(out))
        out = 2.6 * torch.sigmoid(self.linear30(out)) + 0.1
        # out = 3*torch.sigmoid(self.linear6(out))
        return out