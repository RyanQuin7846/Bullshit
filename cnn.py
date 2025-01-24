# -*- coding: utf-8 -*-
"""
Created on Sun Nov  3 19:50:24 2024

@author: Server2
"""

import torch
import torch.nn as nn
import torch.nn.init as init

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.act2 = nn.Softplus()

        # First convolutional layer, input channels = 2 (from x and y), output channels = 64, kernel size = 3
        self.conv1 = nn.Conv2d(in_channels=2, out_channels=64, kernel_size=3, padding=1)
        
        # Define 28 intermediate convolutional layers, with 64 input and 64 output channels
        self.middle_layers = nn.ModuleList([nn.Conv2d(64, 64, kernel_size=3, padding=1) for _ in range(28)])
        
        # Final convolutional layer to output a single channel
        self.conv_last = nn.Conv2d(64, 1, kernel_size=3, padding=1)

        # Initialize parameters
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.uniform_(m.weight, a=-2, b=2)
                init.constant_(m.bias, 0)

    def forward(self, x, y):
        # Concatenate x and y along the channel dimension (assuming x and y are each [batch_size, 1, H, W])
        z = torch.cat((x, y), dim=0)

        # Pass through the first convolutional layer
        out = torch.sigmoid(self.conv1(z))

        # Pass through the 28 intermediate convolutional layers
        for layer in self.middle_layers:
            out = torch.sigmoid(layer(out))

        # Pass through the last convolutional layer
        out = self.conv_last(out)

        return out

