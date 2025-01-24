# -*- coding: utf-8 -*-
"""
Created on Sun Nov  3 19:52:15 2024

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
from cnn import CNN
# from Dnetworks2 import FCNN2
from SG_filter import SG_filter
from torch_01normalize import torch_01normalize

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

# mat_lap, mat_Lx, mat_Ly = SG_filter(phase, y, x)
# laplacian = mat_lap
# gradient_x = mat_Lx + mat_Ly


# stab = stab_ept(y, x, phase, gt, 0.01)

x = x[1:-1,1:-1]
y = y[1:-1,1:-1]
phase = phase[1:-1,1:-1]
gt = gt[1:-1,1:-1]
# pd_s = pd_s[1:-1,1:-1]
gradient_x = gradient_x[1:-1,1:-1]
gradient_y = gradient_y[1:-1,1:-1]
laplacian = laplacian[1:-1,1:-1]
stab = stab[1:-1,1:-1]

plt.clf()
plt.imshow(gt,clim=(0,3))
# plt.title("Best NRMSE\nEPOCH=%d\nNRMSE=%1.3f\nSSIM=%1.3f"%((epoch+1),n,s),fontproperties = 'Times New Roman',fontsize = 25)
plt.xticks([])
plt.yticks([])
plt.colorbar()
plt.savefig("result/figure/gt.jpg",bbox_inches='tight')

loss_list = []
p_g_p_x_list = []
p_g_p_y_list = []
p2_g_list = []

for seed in range(1): 
    n_best = 100
    s_best = -100
    set_seed(seed)
    
    mse_loss = nn.MSELoss()
    
    net = CNN()
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-4, betas=(0.5, 0.999))
    
    cuda = True if torch.cuda.is_available() else False
    
    if cuda:
        net.cuda()
        mse_loss.cuda()
    
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    
    phase = torch.tensor(phase .reshape(1,34,34), requires_grad=True).type(Tensor)
    gradient_x = torch.tensor(gradient_x .reshape(1,34,34), requires_grad=True).type(Tensor)
    gradient_y = torch.tensor(gradient_y .reshape(1,34,34), requires_grad=True).type(Tensor)
    laplacian = torch.tensor(laplacian .reshape(1,34,34), requires_grad=True).type(Tensor)
    gt_t = torch.tensor(gt, requires_grad=True).type(Tensor)
    x = torch.tensor(x .reshape(1,34,34), requires_grad=True).type(Tensor)
    y = torch.tensor(y .reshape(1,34,34), requires_grad=True).type(Tensor)
    
    std = laplacian/2*(2*math.pi*(127.8e6) * (4e-7)*math.pi)
    
    epochs = 100000

    for epoch in range(epochs):
        print(epoch)
        e = net(x,y)
        # e = net(x,y,phase,gradient_x,gradient_y,laplacian)
        gamma = 1/e
        #Compute gradient of gamma
        gamma = gamma.reshape(gt.shape[0],gt.shape[1])
        
        p_g_p_x = torch.autograd.grad(gamma.sum(), x, create_graph=True, retain_graph=True)[0].reshape(gt.shape[0],gt.shape[1])
        p_g_p_y = torch.autograd.grad(gamma.sum(), y, create_graph=True, retain_graph=True)[0].reshape(gt.shape[0],gt.shape[1])
        p2_g_p_x2 = torch.autograd.grad(p_g_p_x.sum(), x, create_graph=True, retain_graph=True)[0].reshape(gt.shape[0],gt.shape[1])
        p2_g_p_y2 = torch.autograd.grad(p_g_p_y.sum(), y, create_graph=True, retain_graph=True)[0].reshape(gt.shape[0],gt.shape[1])
        # p_g = (p_g_p_x + p_g_p_y)
        p2_g = p2_g_p_x2 + p2_g_p_y2
        
        mse = mse_loss(e, torch.tensor(stab).type(Tensor) .reshape(1,34,34))
        
        std_loss = ((gamma * laplacian.reshape(gt.shape[0],gt.shape[1]))
                    - 2*(2*math.pi*(127.8e6) * (4e-7)*math.pi)) + 53722.3242
        
        cr_loss = ((p_g_p_x * gradient_x.reshape(gt.shape[0],gt.shape[1])+p_g_p_y * gradient_y.reshape(gt.shape[0],gt.shape[1]))
                        +(gamma * laplacian.reshape(gt.shape[0],gt.shape[1]))
                        - 2*(2*math.pi*(127.8e6) * (4e-7)*math.pi)) + 2592523.7500
        
        stab_loss = (-0.01*p2_g.reshape(gt.shape[0],gt.shape[1])+(p_g_p_x * gradient_x.reshape(gt.shape[0],gt.shape[1])+p_g_p_y * gradient_y.reshape(gt.shape[0],gt.shape[1]))
                        +(gamma * laplacian.reshape(gt.shape[0],gt.shape[1]))
                        - 2*(2*math.pi*(127.8e6) * (4e-7)*math.pi)) + 21979974
        
        # if epoch < 1e4:
        #     loss = mse
        # elif epoch < 1e5:
        #     loss = mse * 1e2 + torch.norm(std_loss,p=2)
        #     print("Std_loss:%1.3f"%(loss.detach().cpu().numpy()))
        # elif epoch < 2e5:
        #     loss = mse * 1e2 + torch.norm(cr_loss,p=2)
        #     print("CR_loss:%1.3f"%(loss.detach().cpu().numpy()))
        # else:
        #     loss = mse * 1e2 + torch.norm(stab_loss,p=2)
        #     print("Stab_loss:%1.3f"%(loss.detach().cpu().numpy()))
        # loss = mse
        # print(torch.norm(std_loss,p=2))ArithmeticError
        if epoch<1e5:
            loss = mse
        elif epoch<2e5:
            loss = torch.norm(cr_loss,p=2)
        else:
            loss = torch.norm(stab_loss,p=2)
        print(torch.norm(std_loss,p=2))
        print(torch.norm(cr_loss,p=2))
        print(torch.norm(stab_loss,p=2))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        e = e.detach().cpu().numpy()
        e = e.reshape(gt.shape[0],gt.shape[1])
        n = nrmse(e,gt)
        s = ssim(e,gt,data_range=gt.max()-gt.min())
        # 在 epoch=1e5±100 范围内存储所需数据
        # if 1e5 - 100 <= epoch <= 1e5 + 100:
        loss_list.append(loss.detach().cpu().numpy())
        p_g_p_x_list.append(p_g_p_x.detach().cpu().numpy())
        p_g_p_y_list.append(p_g_p_y.detach().cpu().numpy())
        p2_g_list.append(p2_g.detach().cpu().numpy())
        
        # 在 epoch=1e5+100 后保存所有数据为 npz 文件
        # if epoch == 1e5 + 100:
    
        if epoch%10 == 0:
        # if epoch%10 == 0:
            # if loss.detach().cpu().numpy() < 3e4:
            n_best = n
            plt.clf()
            plt.imshow(e)
            plt.title("EPOCH=%d\nNRMSE=%1.3f\nSSIM=%1.3f"%((epoch+1),n,s),fontproperties = 'Times New Roman',fontsize = 25)
            plt.xticks([])
            plt.yticks([])
            plt.colorbar()
            plt.savefig("E:/Qin/PINN/result/figure/seed%d.jpg"%(seed),bbox_inches='tight')
            
            plt.clf()
            plt.imshow(abs(gt-e),cmap='Reds')
            # plt.title("EPOCH=%d\nNRMSE=%1.3f\nSSIM=%1.3f"%((epoch+1),n,s),fontproperties = 'Times New Roman',fontsize = 25)
            plt.xticks([])
            plt.yticks([])
            plt.colorbar()
            plt.savefig("E:/Qin/PINN/result/figure/seed%derrormap.jpg"%(seed),bbox_inches='tight')
                
            if n < n_best:
                n_best = n
                plt.clf()
                plt.imshow(e)
                plt.title("Best NRMSE\nEPOCH=%d\nNRMSE=%1.3f\nSSIM=%1.3f"%((epoch+1),n,s),fontproperties = 'Times New Roman',fontsize = 25)
                plt.xticks([])
                plt.yticks([])
                plt.colorbar()
                plt.savefig("E:/Qin/PINN/result/figure/seed%d_Best NRMSE.jpg"%(seed),bbox_inches='tight')
                
            if s > s_best:
                s_best = s
                plt.clf()
                plt.imshow(e)
                plt.title("Best SSIM\nEPOCH=%d\nNRMSE=%1.3f\nSSIM=%1.3f"%((epoch+1),n,s),fontproperties = 'Times New Roman',fontsize = 25)
                plt.xticks([])
                plt.yticks([])
                plt.colorbar()
                plt.savefig("E:/Qin/PINN/result/figure/seed%d_Best SSIM.jpg"%(seed),bbox_inches='tight')

    np.savez("E:/Qin/PINN/result/data_epoch1e5_plus_minus_100.npz",
             loss=loss_list,
             p_g_p_x=p_g_p_x_list,
             p_g_p_y=p_g_p_y_list,
             p2_g=p2_g_list)