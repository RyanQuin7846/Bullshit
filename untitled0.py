# -*- coding: utf-8 -*-
"""
Created on Mon Sep  2 15:21:58 2024

@author: Server2
"""

import numpy as np
import math
from SG_filter import SG_filter
import matplotlib.pyplot as plt

# 输入数据 (x, y, z)
matrix_x = np.load("data/raw data/4mm tumor pos1/X.npy")
x = matrix_x[0].T
matrix_y = np.load("data/raw data/4mm tumor pos1/Y.npy")
y = matrix_y[0].T
phase = np.load("data/raw data/4mm tumor pos1/H.npy")
# phase = phase[0].imag.T
phH1p = np.zeros((1,phase.shape[0],phase.shape[1],phase.shape[2]))
for mm in range(2):
    phH1p[:,mm,:] = np.unwrap(np.angle(phase[mm,:,:],deg=False))
phase = phH1p[0,0].T
gt = np.load("data/raw data/4mm tumor pos1/condy.npy")
gt = gt[0].T
gamma = 1/gt
omega = 2*math.pi*127.8e6

mat_lap, mat_Lx, mat_Ly = SG_filter(phase, x, y)
mat_c_lap, mat_c_Lx, mat_c_Ly = SG_filter(gamma, x, y)
laplacian = mat_lap
gradient = mat_Lx + mat_Ly
c_laplacian = mat_c_lap
c_gradient = mat_c_Lx + mat_c_Ly

x = x[1:-1,1:-1]
y = y[1:-1,1:-1]
phase = phase[1:-1,1:-1]
gt = gt[1:-1,1:-1]
gamma = gamma[1:-1,1:-1]

# residual = (-0.00017583963926881552 * c_laplacian + 0.1169634610414505*(c_gradient * gradient)
#                 +0.2824136018753052* (gamma * laplacian)
#                 - (2*math.pi*(127.8e6) * (4e-7)*math.pi))

residual = (-0.00024511985247954726 * c_laplacian + 0.14142243564128876*(c_gradient * gradient)
                +0.3079729974269867*(gamma * laplacian)
                - (2*math.pi*(127.8e6) * (4e-7)*math.pi))

r_m = np.sqrt(np.sum(residual**2))

plt.clf()
plt.imshow(gt,clim=(0,3))
# plt.title("Best NRMSE\nEPOCH=%d\nNRMSE=%1.3f\nSSIM=%1.3f"%((epoch+1),n,s),fontproperties = 'Times New Roman',fontsize = 25)
plt.xticks([])
plt.yticks([])
plt.colorbar()
plt.savefig("result/figure/gt.jpg",bbox_inches='tight')