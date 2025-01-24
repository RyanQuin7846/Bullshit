# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 18:59:25 2024

@author: Server2
"""

import torch

def torch_normalize_matrix(data, min_value=0.4, max_value=0.9):
    
    mask = data != 0
    
    # Calculate min and max values ignoring zeros
    data_min = torch.min(data[mask]).float()
    data_max = torch.max(data[mask]).float()
    
    # Normalize non-zero values to range 0 to 1
    normalized_data = torch.zeros_like(data, dtype=torch.float32)
    normalized_data[mask] = (data[mask].float() - data_min) / (data_max - data_min)
    
    # Scale non-zero values to range min_value to max_value
    scaled_data = torch.zeros_like(data, dtype=torch.float32)
    scaled_data[mask] = normalized_data[mask] * (max_value - min_value) + min_value
    
    return scaled_data