# -*- coding: utf-8 -*-
"""
Created on Tue Jan  7 16:25:42 2025

@author: Server2
"""

import numpy as np
import torch.nn as nn
import torch
import matplotlib.pyplot as plt
import math
from skimage.metrics import normalized_root_mse as nrmse
from skimage.metrics import structural_similarity as ssim

from data_maker import data_maker
from set_seed import set_seed
from Dnetworks import FCNN


y, x, phase, gradient_y, gradient_x, laplacian, stab, gt = data_maker(0.005)
_, _, _, _, _, _, cr, _ = data_maker(0)

phase = phase[1:-1,1:-1]
gt = gt[1:-1,1:-1]
x = x[1:-1,1:-1]
y = y[1:-1,1:-1]

omega = 2*math.pi*127.8e6
mu_0 = (4e-7)*math.pi
n_best = 100
s_best = -100

std = laplacian/(2*(2*math.pi*(127.8e6) * (4e-7)*math.pi))

x = x[1:-1,1:-1]
y = y[1:-1,1:-1]
gradient_x = gradient_x[1:-1,1:-1]
gradient_y = gradient_y[1:-1,1:-1]
laplacian = laplacian[1:-1,1:-1]
std = std[1:-1,1:-1]
cr = cr[1:-1,1:-1]
stab = stab[1:-1,1:-1]
gt = gt[1:-1,1:-1]

plt.clf()
plt.imshow(gt,clim=(0,3),interpolation='bilinear')
plt.xticks([])
plt.yticks([])
plt.colorbar()
plt.savefig("result/figure/gt.jpg",bbox_inches='tight')

loss_list = []
ssim_list = []
ssim_std_list = []
ssim_cr_list = []
ssim_stab_list = []

seed = 0

n_best = 100
s_best = -100
set_seed(seed)

mse_loss = nn.MSELoss()

net = FCNN()
optimizer = torch.optim.Adam(net.parameters(), lr=1e-5, betas=(0.5, 0.999))

cuda = True if torch.cuda.is_available() else False

if cuda:
    net.cuda()
    mse_loss.cuda()

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

phase = torch.tensor(phase.reshape(-1,1), requires_grad=True).type(Tensor)
gradient_x = torch.tensor(gradient_x.reshape(-1,1), requires_grad=True).type(Tensor)
gradient_y = torch.tensor(gradient_y.reshape(-1,1), requires_grad=True).type(Tensor)
laplacian = torch.tensor(laplacian.reshape(-1,1), requires_grad=True).type(Tensor)
x = torch.tensor(x.reshape(-1,1), requires_grad=True).type(Tensor)
y = torch.tensor(y.reshape(-1,1), requires_grad=True).type(Tensor)


epochs = 500000

for epoch in range(epochs):
    print(epoch)
    e = net(x,y)
        
    if epoch < 1e1:
        mse = mse_loss(e, torch.tensor(stab).type(Tensor).reshape(-1,1))
        loss = mse
    else:
        gamma = 1/e
        gamma = gamma.reshape(gt.shape[0],gt.shape[1])
        
        p_g_p_x = torch.autograd.grad(gamma.sum(), x, create_graph=True, retain_graph=True)[0].reshape(gt.shape[0],gt.shape[1])
        p_g_p_y = torch.autograd.grad(gamma.sum(), y, create_graph=True, retain_graph=True)[0].reshape(gt.shape[0],gt.shape[1])
        p2_g_p_x2 = torch.autograd.grad(p_g_p_x.sum(), x, create_graph=True, retain_graph=True)[0].reshape(gt.shape[0],gt.shape[1])
        p2_g_p_y2 = torch.autograd.grad(p_g_p_y.sum(), y, create_graph=True, retain_graph=True)[0].reshape(gt.shape[0],gt.shape[1])
        
        p2_g = p2_g_p_x2 + p2_g_p_y2

        cr_loss = ((p_g_p_x * gradient_x.reshape(gt.shape[0],gt.shape[1])+p_g_p_y * gradient_y.reshape(gt.shape[0],gt.shape[1]))
                        +(gamma * laplacian.reshape(gt.shape[0],gt.shape[1]))
                        - 2*omega*mu_0)
        if epoch == 1e1:
            cr_loss_previous = cr_loss.detach()
            annealing = 2*omega*mu_0

        annealing = annealing * (1+(torch.norm(cr_loss,p=2).detach()-torch.norm(cr_loss_previous,p=2))/torch.norm(cr_loss,p=2).detach())
        cr_loss_previous = cr_loss.detach()
        loss = torch.norm(cr_loss+annealing,p=2)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    e = e.detach().cpu().numpy()
    e = e.reshape(gt.shape[0],gt.shape[1])
    n = nrmse(e,gt)
    s = ssim(e,gt,data_range=gt.max()-gt.min())
    s_std = ssim(e,std,data_range=gt.max()-gt.min())
    s_cr = ssim(e,cr,data_range=gt.max()-gt.min())
    s_stab = ssim(e,stab,data_range=gt.max()-gt.min())
    loss_list.append(loss.detach().cpu().numpy())
    ssim_list.append(s)
    ssim_std_list.append(s_std)
    ssim_cr_list.append(s_cr)
    ssim_stab_list.append(s_stab)
    
    if (epoch+1)%100 == 0:
        n_best = n
        plt.clf()
        plt.imshow(e,clim=(0,3),interpolation='bilinear')
        plt.title("EPOCH=%d\nNRMSE=%1.3f\nSSIM=%1.3f"%((epoch+1),n,s),fontproperties = 'Times New Roman',fontsize = 25)
        plt.xticks([])
        plt.yticks([])
        plt.colorbar()
        plt.savefig("E:/Qin/PINN/result/figure/seed%d.jpg"%(seed),bbox_inches='tight')
        
        plt.clf()
        plt.imshow(abs(gt-e),interpolation='bilinear',cmap='Reds')
        plt.xticks([])
        plt.yticks([])
        plt.colorbar()
        plt.savefig("E:/Qin/PINN/result/figure/seed%derrormap.jpg"%(seed),bbox_inches='tight')
            
            
        if s > s_best:
            s_best = s
            plt.clf()
            plt.imshow(e,clim=(0,3),interpolation='bilinear')
            plt.title("Best SSIM\nEPOCH=%d\nNRMSE=%1.3f\nSSIM=%1.3f"%((epoch+1),n,s),fontproperties = 'Times New Roman',fontsize = 25)
            plt.xticks([])
            plt.yticks([])
            plt.colorbar()
            plt.savefig("E:/Qin/PINN/result/figure/seed%d_Best SSIM.jpg"%(seed),bbox_inches='tight')
            
            plt.clf()
            plt.imshow(abs(gt-e),interpolation='bilinear',cmap='Reds')
            plt.xticks([])
            plt.yticks([])
            plt.colorbar()
            plt.savefig("E:/Qin/PINN/result/figure/seed%d_Best SSIM_errormap.jpg"%(seed),bbox_inches='tight')

np.savez("E:/Qin/PINN/result/data_epoch1e5_plus_minus_100.npz",
         loss=loss_list,
         s=ssim_list,
         s_std=ssim_std_list,
         s_cr=ssim_cr_list,
         s_stab=ssim_stab_list)