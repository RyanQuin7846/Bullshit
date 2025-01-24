# -*- coding: utf-8 -*-
"""
Created on Sat Aug  3 17:14:02 2024

@author: Server2
"""

import numpy as np

def normalize_matrix(data, min_value=0.4, max_value=0.9):
    
    mask = data != 0
    
    # Calculate min and max values ignoring zeros
    data_min = np.min(data[mask])
    data_max = np.max(data[mask])
    
    # Normalize non-zero values to range 0 to 1
    normalized_data = np.zeros_like(data, dtype=float)
    normalized_data[mask] = (data[mask] - data_min) / (data_max - data_min)
    
    # Scale non-zero values to range min_value to max_value
    scaled_data = np.zeros_like(data, dtype=float)
    scaled_data[mask] = normalized_data[mask] * (max_value - min_value) + min_value
    
    return scaled_data


gt = np.load("data/raw data/no tumor/condy.npy")

pd = normalize_matrix(gt)