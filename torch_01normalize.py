# -*- coding: utf-8 -*-
"""
Created on Thu Aug 29 17:18:35 2024

@author: Server2
"""

import torch

def torch_01normalize(data):
    
    mask = data != 0
    
    # Calculate min and max values ignoring zeros
    data_min = torch.min(data[mask]).float()
    data_max = torch.max(data[mask]).float()
    
    # Normalize non-zero values to range 0 to 1
    normalized_data = torch.zeros_like(data, dtype=torch.float32)
    normalized_data[mask] = (data[mask].float() - data_min) / (data_max - data_min)

    
    return normalized_data