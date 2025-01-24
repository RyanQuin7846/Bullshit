# -*- coding: utf-8 -*-
"""
Created on Thu Jan  2 16:33:59 2025

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

import pywt

rho = 0.000

mode = 'PINN'
if mode == 'PINN':
    y, x, phase, gradient_y, gradient_x, laplacian, stab, gt = data_maker(rho)
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
    
elif mode == 'no tumor':
    matrix_x = np.load("data/raw data/no tumor/X.npy")
    x = matrix_x[0].T
    matrix_y = np.load("data/raw data/no tumor/Y.npy")
    y = matrix_y[0].T
    phase = np.load("data/raw data/no tumor/H.npy")
    phase = phase[0].imag.T
    gt = np.load("data/raw data/no tumor/condy.npy")
    gt = gt[0].T
    
elif mode == '4mm tumor pos1':
    matrix_x = np.load("data/raw data/4mm tumor pos1/X.npy")
    x = matrix_x[0]
    matrix_y = np.load("data/raw data/4mm tumor pos1/Y.npy")
    y = matrix_y[0]
    phase = np.load("data/raw data/4mm tumor pos1/H.npy")
    phH1p = np.zeros((1,phase.shape[0],phase.shape[1],phase.shape[2]))
    for mm in range(2):
        phH1p[:,mm,:] = np.unwrap(np.angle(phase[mm,:,:],deg=False))
    phase = phH1p[0,0]
    gt = np.load("data/raw data/4mm tumor pos1/condy.npy")
    gt = gt[0]
    pd_s = normalize_matrix(np.load("data/raw data/no tumor/condy.npy")[0])

else:
    print("Error: No data has been read")
    sys.exit()


omega = 2*math.pi*127.8e6
n_best = 100
s_best = -100

mat_lap, mat_Lx, mat_Ly = SG_filter(phase, y, x)
# laplacian = mat_lap
# gradient_x = mat_Lx + mat_Ly
std = laplacian/((math.pi*(127.8e6) * (4e-7)*math.pi))
# stab = std1
# stab = stab_ept(y, x, phase, gt, 0.01)
cr = stab_ept(y, x, phase, gt, 0)
# stab = gt

# stab = np.round(stab,decimals=3)

laplacian_stab, gradient_stab_x, gradient_stab_y = SG_filter(stab, y, x)

gradient_stab_x = np.zeros([34,34])
gradient_stab_y = np.zeros([34,34])

for i in range(1,35):
    for j in range(1,35):
        gradient_stab_x[j-1,i-1] = (stab[j,i+1] - stab[j,i-1])/(2*2e-3)
        gradient_stab_y[j-1,i-1] = (stab[j+1,i] - stab[j-1,i])/(2*2e-3)
        
mean_x = np.mean(gradient_stab_x)
mean_y = np.mean(gradient_stab_y)
sd_x = np.std(gradient_stab_x)
sd_y = np.std(gradient_stab_y)

threshold_x = np.percentile(abs(gradient_stab_x), (0.341)*100)

threshold_y = np.percentile(abs(gradient_stab_y), (0.341)*100)

indices = np.where((abs(gradient_stab_x) <= threshold_x) & (abs(gradient_stab_y) <= threshold_y))

true_false_matrix = ((abs(gradient_stab_x) <= threshold_x) & (abs(gradient_stab_y) <= threshold_y))

x = x[1:-1,1:-1]
y = y[1:-1,1:-1]
phase = phase[1:-1,1:-1]
gt = gt[1:-1,1:-1]
# pd_s = pd_s[1:-1,1:-1]
gradient_x = gradient_x[1:-1,1:-1]
gradient_y = gradient_y[1:-1,1:-1]
laplacian = laplacian[1:-1,1:-1]
stab = stab[1:-1,1:-1]
std = laplacian/(2*(2*math.pi*(127.8e6) * (4e-7)*math.pi))
# cr = stab_ept(y, x, phase, gt, 0
stab= std
a = gradient_stab_x.reshape(1,-1)[0]
b = gradient_stab_y.reshape(1,-1)[0]

# plt.cla()
# plt.imshow(gt,clim=(0,3))
# plt.xticks([])
# plt.yticks([])
# plt.colorbar()
# plt.show()

# plt.cla()
# plt.imshow(true_false_matrix, cmap=('gray_r'))
# plt.xticks([])
# plt.yticks([])
# plt.colorbar(ticks=[0, 1]).ax.set_yticklabels(['False', 'True'])
# plt.show()

# plt.cla()
# plt.imshow(gradient_stab_x,clim=(np.mean(gradient_stab_x)-3*np.std(gradient_stab_x),np.mean(gradient_stab_x)+3*np.std(gradient_stab_x)))
# plt.xticks([])
# plt.yticks([])
# plt.colorbar()
# plt.show()

# plt.cla()
# plt.imshow(gradient_stab_y,clim=(np.mean(gradient_stab_y)-3*np.std(gradient_stab_y),np.mean(gradient_stab_y)+3*np.std(gradient_stab_y)))
# plt.xticks([])
# plt.yticks([])
# plt.colorbar()
# plt.show()
c_x = []
c_y = []
for m in range(int(stab.shape[0])):
    c_t = np.corrcoef(stab[m],gt[m])[0,-1]
    c_y.append(c_t)
    plt.clf()
    plt.plot(gt[m],color='black',label='Ground truth')
    plt.plot(stab[m],color='r',label='Stab-EPT')
    plt.ylim(0,3)
    plt.legend()
    # plt.show()
    plt.savefig("E:/Qin/PINN/result/figure/stab+rho=%f1.3+ylineprofile%d.jpg"%(rho,m),bbox_inches='tight')
  
plt.clf()
plt.plot(c_y)
plt.ylim(0,1)
plt.title('Correlation coefficient')
# plt.show()
plt.savefig("E:/Qin/PINN/result/figure/cc_y.jpg",bbox_inches='tight')
    
# for n in range(int(stab.shape[1])):
#     c_t = np.corrcoef(stab[:,n],gt[:,n])[0,-1]
#     c_x.append(c_t)
#     plt.clf()
#     plt.plot(gt[:,n],color='black',label='Ground truth')
#     plt.plot(stab[:,n],color='r',label='Stab-EPT')
#     plt.ylim(0,3)
#     plt.legend()
#     # plt.show()
#     plt.savefig("E:/Qin/PINN/result/figure/stab+rho=%f1.3+xlineprofile%d.jpg"%(rho,n),bbox_inches='tight')



# plt.clf()
# plt.plot(c_x)
# plt.ylim(0,1)
# plt.title('Correlation coefficient')
# # plt.show()
# plt.savefig("E:/Qin/PINN/result/figure/cc_x.jpg",bbox_inches='tight')