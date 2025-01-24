# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 11:40:03 2024

@author: Server2
"""

import numpy as np
import math
from stabEPT import stab_ept
from SG_filter import SG_filter

PHANTOM2_conductivity = np.load("data/circular/0.1-1.0_2PHANTOM_v5Original_conductivity.npy")
PHANTOM2_phase = np.load("data/circular/0.1-1.0_2PHANTOM_v5Original_phase.npy")

PHANTOM4_conductivity = np.load("data/circular/0.5-1.0_4PHANTOM_v5Original_conductivity.npy")
PHANTOM4_phase = np.load("data/circular/0.5-1.0_4PHANTOM_v5Original_phase.npy")

PHANTOM6_conductivity = np.load("data/circular/1.0-0.1_6PHANTOM_v5Original_conductivity.npy")
PHANTOM6_phase = np.load("data/circular/1.0-0.1_6PHANTOM_v5Original_phase.npy")

PHANTOM7_conductivity = np.load("data/circular/1.0-0.5_7PHANTOM_v5Original_conductivity.npy")
PHANTOM7_phase = np.load("data/circular/1.0-0.5_7PHANTOM_v5Original_phase.npy")

PHANTOM8_conductivity = np.load("data/circular/1.0-1.5_8PHANTOM_v5Original_conductivity.npy")
PHANTOM8_phase = np.load("data/circular/1.0-1.5_8PHANTOM_v5Original_phase.npy")

PHANTOM9_conductivity = np.load("data/circular/1.0-2.0_9PHANTOM_v5Original_conductivity.npy")
PHANTOM9_phase = np.load("data/circular/1.0-2.0_9PHANTOM_v5Original_phase.npy")

PHANTOM9_conductivity = np.load("data/circular/1.0-2.0_9PHANTOM_v5Original_conductivity.npy")
PHANTOM9_phase = np.load("data/circular/1.0-2.0_9PHANTOM_v5Original_phase.npy")

PHANTOM9_conductivity = np.load("data/circular/1.0-2.0_9PHANTOM_v5Original_conductivity.npy")
PHANTOM9_phase = np.load("data/circular/1.0-2.0_9PHANTOM_v5Original_phase.npy")

PHANTOM11_conductivity = np.load("data/circular/1.5-1.0_11PHANTOM_v5Original_conductivity.npy")
PHANTOM11_phase = np.load("data/circular/1.5-1.0_11PHANTOM_v5Original_phase.npy")

PHANTOM13_conductivity = np.load("data/circular/2.0-1.0_13PHANTOM_v5Original_conductivity.npy")
PHANTOM13_phase = np.load("data/circular/2.0-1.0_13PHANTOM_v5Original_phase.npy")

# # Define the step size and grid dimensions
# step_size = 0.0018604090881272726
# grid_size = 52

# # Generate the x and y grids
# x = np.linspace(0, (grid_size - 1) * step_size, grid_size)
# y = np.linspace(0, (grid_size - 1) * step_size, grid_size)

# # Create the 2D meshgrid matrices for x and y directions
# X, Y = np.meshgrid(x, y)

tumor_conductivity = np.load("data/raw data/4mm tumor pos1/condy.npy")[0]
tumor_H = np.load("data/raw data/4mm tumor pos1/H.npy")
phH1p = np.zeros((1,tumor_H.shape[0],tumor_H.shape[1],tumor_H.shape[2]))
for mm in range(2):
    phH1p[:,mm,:] = np.unwrap(np.angle(tumor_H[mm,:,:],deg=False))
tumor_phase = phH1p[0,0]
matrix_x = np.load("data/raw data/4mm tumor pos1/X.npy")
y = matrix_x[0]
matrix_y = np.load("data/raw data/4mm tumor pos1/Y.npy")
x = matrix_y[0]

phase = tumor_phase
gt = tumor_conductivity

mat_lap, mat_Lx, mat_Ly = SG_filter(phase, x, y)
stab = stab_ept(x, y, phase, gt, 0.01)
std = mat_lap/(2*math.pi*(127.8e6) * (4e-7)*math.pi)