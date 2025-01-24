# -*- coding: utf-8 -*-
"""
Created on Wed Dec 25 19:52:03 2024

@author: Server2
"""

import sys
import numpy as np
import torch.nn as nn
import torch
import matplotlib.pyplot as plt
import math
from skimage.metrics import normalized_root_mse as nrmse
from skimage.metrics import structural_similarity as ssim
import pandas as pd

from data_maker import data_maker
from normalize_matrix import normalize_matrix
from torch_normalize_matrix import torch_normalize_matrix
from stabEPT import stab_ept
from set_seed import set_seed
# from networks import FCNN
from Dnetworks import FCNN
# from Dnetworks2 import FCNN2
from SG_filter import SG_filter
from torch_01normalize import torch_01normalize
from scipy.ndimage import zoom

mode = 'PINN'
if mode == 'PINN':
    y, x, phase, gradient_y, gradient_x, laplacian, stab, gt = data_maker(0.005)
    # stab2 = data_maker(0.01,'stab')[2:-2,2:-2]
    # stab3 = data_maker(0.015,'stab')[2:-2,2:-2]
    # stab4 = data_maker(0.02,'stab')[2:-2,2:-2]
    # stab5 = data_maker(0.025,'stab')[2:-2,2:-2]
    phase = phase[1:-1,1:-1]
    gt = gt[1:-1,1:-1]
    x = x[1:-1,1:-1]
    y = y[1:-1,1:-1]
    # gradient = gradient_x + gradient_y
    # pd_s = np.load('data/pd_wotumor.npy')

zoom_factor = 5

interpolated_x = zoom(x, zoom_factor, order=1)
interpolated_y = zoom(y, zoom_factor, order=1)
interpolated_phase = zoom(phase, zoom_factor, order=1)
interpolated_gt = zoom(gt, zoom_factor, order=1)

mat_lap, mat_Lx, mat_Ly = SG_filter(interpolated_phase, interpolated_y, interpolated_x)

stab = stab_ept(interpolated_y, interpolated_x, interpolated_phase, interpolated_gt, 0.005)
std = mat_lap/(2*(2*math.pi*(127.8e6) * (4e-7)*math.pi))
