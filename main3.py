# -*- coding: utf-8 -*-
"""
Created on Thu Aug 29 16:56:16 2024

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
from networks import FCNN
from SG_filter import SG_filter

mode = '4mm tumor double'
if mode == 'PINN':
    y, x, phase, gradient_y, gradient_x, laplacian, stab, gt = data_maker(0.005)
    stab2 = data_maker(0.01,'stab')[2:-2,2:-2]
    stab3 = data_maker(0.015,'stab')[2:-2,2:-2]
    stab4 = data_maker(0.02,'stab')[2:-2,2:-2]
    stab5 = data_maker(0.025,'stab')[2:-2,2:-2]
    phase = phase[1:-1,1:-1]
    gt = gt[1:-1,1:-1]
    x = x[3:-3,3:-3]
    y = y[3:-3,3:-3]
    gradient = gradient_x + gradient_y
    pd_s = np.load('data/pd_wotumor.npy')
    
elif mode == 'no tumor':
    matrix_x = np.load("data/raw data/no tumor/X.npy")
    x = matrix_x[0].T
    matrix_y = np.load("data/raw data/no tumor/Y.npy")
    y = matrix_y[0].T
    phase = np.load("data/raw data/no tumor/H.npy")
    phase = phase[0].imag.T
    gt = np.load("data/raw data/no tumor/condy.npy")
    gt = gt[0].T
    
elif mode == '4mm tumor double':
    matrix_x = np.load("data/raw data/4mm tumor double/X.npy")
    x = matrix_x[0].T
    matrix_y = np.load("data/raw data/4mm tumor double/Y.npy")
    y = matrix_y[0].T
    phase = np.load("data/raw data/4mm tumor double/H.npy")
    phase = phase[0].imag.T
    gt = np.load("data/raw data/4mm tumor double/condy.npy")
    gt = gt[0].T
    pd_s = normalize_matrix(np.load("data/raw data/no tumor/condy.npy")[0].T)

else:
    print("Error: No data has been read")
    sys.exit()


omega = 2*math.pi*127.8e6
n_best = 100
s_best = -100

mat_lap, mat_Lx, mat_Ly = SG_filter(phase, x, y)
laplacian = mat_lap
gradient = mat_Lx + mat_Ly

stab1 = stab_ept(x, y, phase, gt, 0.01)
stab2 = stab_ept(x, y, phase, gt, 0.015)
stab3 = stab_ept(x, y, phase, gt, 0.02)
stab4 = stab_ept(x, y, phase, gt, 0.025)
stab5 = stab_ept(x, y, phase, gt, 0.023)


x = x[1:-1,1:-1]
y = y[1:-1,1:-1]
phase = phase[1:-1,1:-1]
gt = gt[1:-1,1:-1]
pd_s = pd_s[1:-1,1:-1]

for seed in range(10): 
    n_best = 100
    s_best = -100
    set_seed(seed)
    
    net = FCNN()
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3, betas=(0.5, 0.999))
    mse_loss = nn.MSELoss()
    
    cuda = True if torch.cuda.is_available() else False
    
    if cuda:
        net.cuda()
        mse_loss.cuda()
    
    record = []
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    phase = torch.tensor(phase.reshape(-1,1), requires_grad=True).type(Tensor)
    gradient = torch.tensor(gradient.reshape(-1,1), requires_grad=True).type(Tensor)
    laplacian = torch.tensor(laplacian.reshape(-1,1), requires_grad=True).type(Tensor)
    x = torch.tensor(x.reshape(-1,1), requires_grad=True).type(Tensor)
    y = torch.tensor(y.reshape(-1,1), requires_grad=True).type(Tensor)
    pd_s = torch.tensor(pd_s, requires_grad=True).type(Tensor)
    
    epochs = 50000

    for epoch in range(epochs):
        print(epoch)
        e = net(torch.tensor(stab1.reshape(-1,1)).type(Tensor),torch.tensor(stab2.reshape(-1,1)).type(Tensor),torch.tensor(stab3.reshape(-1,1)).type(Tensor),torch.tensor(stab4.reshape(-1,1)).type(Tensor),torch.tensor(stab5.reshape(-1,1)).type(Tensor),)
        loss = mse_loss(e, torch.tensor(gt.reshape(-1,1)).type(Tensor))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        e = e.detach().cpu().numpy()
        e = e.reshape(gt.shape[0],gt.shape[1])
        n = nrmse(e,gt)
        s = ssim(e,gt,data_range=gt.max()-gt.min())
        # if (epoch+1)%100==0:
        # plt.clf()
        # plt.imshow(e,clim=(0,3))
        # plt.title("EPOCH=%d\nNRMSE=%1.3f\nSSIM=%1.3f"%((epoch+1),n,s),fontproperties = 'Times New Roman',fontsize = 25)
        # plt.xticks([])
        # plt.yticks([])
        # plt.colorbar()
        # plt.savefig("result/figure/%d_network_c epochs.jpg"%(epoch+1),bbox_inches='tight')
            
        if n < n_best:
            n_best = n
            plt.clf()
            plt.imshow(e,clim=(0,3))
            plt.title("Best NRMSE\nEPOCH=%d\nNRMSE=%1.3f\nSSIM=%1.3f"%((epoch+1),n,s),fontproperties = 'Times New Roman',fontsize = 25)
            plt.xticks([])
            plt.yticks([])
            plt.colorbar()
            plt.savefig("result/figure/seed%d_Best NRMSE.jpg"%(seed),bbox_inches='tight')
            
        if s > s_best:
            s_best = s
            plt.clf()
            plt.imshow(e,clim=(0,3))
            plt.title("Best SSIM\nEPOCH=%d\nNRMSE=%1.3f\nSSIM=%1.3f"%((epoch+1),n,s),fontproperties = 'Times New Roman',fontsize = 25)
            plt.xticks([])
            plt.yticks([])
            plt.colorbar()
            plt.savefig("result/figure/seed%d_Best SSIM.jpg"%(seed),bbox_inches='tight')   