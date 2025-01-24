# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 17:48:57 2024

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
from SG_filter import SG_filter
from torch_01normalize import torch_01normalize

mode = '4mm tumor pos1'
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
    gradient = gradient_x + gradient_y
    pd_s = np.load('data/pd_wotumor.npy')
    
elif mode == '4mm tumor pos1':
    matrix_x = np.load("data/raw data/4mm tumor pos1/X.npy")
    x = matrix_x[0].T
    matrix_y = np.load("data/raw data/4mm tumor pos1/Y.npy")
    y = matrix_y[0].T
    phase = np.load("data/raw data/4mm tumor pos1/H.npy")
    phase = phase[0].imag.T
    gt = np.load("data/raw data/4mm tumor pos1/condy.npy")
    gt = gt[0].T
    
elif mode == '20240908':
    matrix_x = np.load("data/raw data/20240908/X.npy")
    x = matrix_x[0].T
    matrix_y = np.load("data/raw data/20240908/Y.npy")
    y = matrix_y[0].T
    phase = np.load("data/raw data/20240908/H.npy")
    # phH1p = np.zeros((1,phase.shape[0],phase.shape[1],phase.shape[2]))
    # for mm in range(2):
    #     phH1p[:,mm,:] = np.unwrap(np.angle(phase[mm,:,:],deg=False))
    # phase = phH1p[0,0].T
    phase = phase.imag.T
    gt = np.load("data/raw data/20240908/condy.npy")
    gt = gt[0].T
    pd_s = normalize_matrix(np.load("data/raw data/4mm tumor pos1/condy.npy")[0])

else:
    print("Error: No data has been read")
    sys.exit()


omega = 2*math.pi*127.8e6
n_best = 100
s_best = -100

mat_lap, mat_Lx, mat_Ly = SG_filter(phase, x, y)
laplacian = mat_lap
gradient = mat_Lx + mat_Ly


stab = stab_ept(x, y, phase, gt, 0.01)

x = x[1:-1,1:-1]
y = y[1:-1,1:-1]
phase = phase[1:-1,1:-1]
gt = gt[1:-1,1:-1]
pd_s = pd_s[1:-1,1:-1]

plt.clf()
plt.imshow(gt,clim=(0,3))
# plt.title("Best NRMSE\nEPOCH=%d\nNRMSE=%1.3f\nSSIM=%1.3f"%((epoch+1),n,s),fontproperties = 'Times New Roman',fontsize = 25)
plt.xticks([])
plt.yticks([])
plt.colorbar()
plt.savefig("result/figure/gt.jpg",bbox_inches='tight')

for seed in range(1): 
    n_best = 100
    s_best = -100
    set_seed(seed)
    
    net = FCNN()
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-4, betas=(0.5, 0.999))
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
    gt_t = torch.tensor(gt, requires_grad=True).type(Tensor)
    x = torch.tensor(x.reshape(-1,1), requires_grad=True).type(Tensor)
    y = torch.tensor(y.reshape(-1,1), requires_grad=True).type(Tensor)
    pd_s = torch.tensor(pd_s, requires_grad=True).type(Tensor)
    
    random1 = torch.rand(1892,1)
    random2 = torch.rand(1892,1)
    random3 = torch.rand(1892,1)
    
    epochs = 500000

    for epoch in range(epochs):
        print(epoch)
        # e = net(x,y,phase,gradient,laplacian)
        e = net(x,y,torch_01normalize(phase),torch_01normalize(gradient),torch_01normalize(laplacian))
        # e = net(x,y)
        mse = mse_loss(e, torch.tensor(stab).type(Tensor).reshape(-1,1))
        gamma = 1/e
        #Compute gradient of gamma
        gamma = gamma.reshape(gt.shape[0],gt.shape[1])
        
        p_g_p_x = torch.autograd.grad(gamma.sum(), x, create_graph=True, retain_graph=True)[0].reshape(gt.shape[0],gt.shape[1])
        p_g_p_y = torch.autograd.grad(gamma.sum(), y, create_graph=True, retain_graph=True)[0].reshape(gt.shape[0],gt.shape[1])
        p2_g_p_x2 = torch.autograd.grad(p_g_p_x.sum(), x, create_graph=True, retain_graph=True)[0].reshape(gt.shape[0],gt.shape[1])
        p2_g_p_y2 = torch.autograd.grad(p_g_p_y.sum(), y, create_graph=True, retain_graph=True)[0].reshape(gt.shape[0],gt.shape[1])
        p_g = (p_g_p_x + p_g_p_y)
        p2_g = p2_g_p_x2 + p2_g_p_y2

        # pde_loss_2d = (-0.01*p2_g.reshape(gt.shape[0],gt.shape[1])+(p_g * gradient.reshape(gt.shape[0],gt.shape[1]))
        #                 +(gamma * laplacian.reshape(gt.shape[0],gt.shape[1]))
        #                 - (2*math.pi*(127.8e6) * (4e-7)*math.pi))
        
        # pde_loss_2d = ((gamma * laplacian.reshape(gt.shape[0],gt.shape[1]))
        #                 - (2*math.pi*(127.8e6) * (4e-7)*math.pi))

        pde_loss_2d = ((p_g * gradient.reshape(gt.shape[0],gt.shape[1]))
                        +(gamma * laplacian.reshape(gt.shape[0],gt.shape[1]))
                        - (2*math.pi*(127.8e6) * (4e-7)*math.pi))

        # ss = ssim_loss(torch_normalize_matrix(e.reshape(1,1,gt.shape[0],gt.shape[1])),pd_s.reshape(1,1,gt.shape[0],gt.shape[1]))
        # sm = mse_loss(torch_normalize_matrix(e.reshape(1,1,gt.shape[0],gt.shape[1])),pd_s.reshape(1,1,gt.shape[0],gt.shape[1]))
        
        # if epoch% 5 == 0:
        # loss = torch.log(torch.norm(pde_loss_2d,p=2)) + torch.log(torch.norm(pde_loss_2d,p=1))
        # if epoch < 1e4:
        #     loss = mse
        # else:
        e_m = e.reshape(gt.shape[0],gt.shape[1])
        b_loss = mse_loss(e_m[0],gt_t[0]) + mse_loss(e_m[-1],gt_t[-1]) + mse_loss(e_m[:,0],gt_t[:,0]) + mse_loss(e_m[:,-1],gt_t[:,-1])
        loss = torch.log(torch.norm(pde_loss_2d,p=2)) + torch.log(torch.norm(pde_loss_2d,p=1)) + 10* b_loss
        # print((torch.norm(pde_loss_2d,p=2)))
        #     # loss = mse + 0.0001*torch.log(torch.norm(pde_loss_2d,p=2)) + torch.log(torch.norm(pde_loss_2d,p=1))
        print(mse,torch.log(torch.norm(pde_loss_2d,p=2)),torch.log(torch.norm(pde_loss_2d,p=1)),b_loss)
        # else:
        # loss = 250*sm
        #     print(sm)
            
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        e = e.detach().cpu().numpy()
        e = e.reshape(gt.shape[0],gt.shape[1])
        n = nrmse(e,gt)
        s = ssim(e,gt,data_range=gt.max()-gt.min())
        record.append({
        'Epoch': epoch + 1,
        'PDE L2': torch.norm(pde_loss_2d,p=2).item(),
        'PDE L1': torch.norm(pde_loss_2d,p=1).item(),
        'Loss': loss.item(),
        'rho*gamma': (torch.norm(-0.05 * p2_g.reshape(gt.shape[0],gt.shape[1]),p=2)).item(),
        'g*l':(torch.norm(gamma * laplacian.reshape(gt.shape[0],gt.shape[1]),p=2)).item(),
        'g*g':(torch.norm(p_g * gradient.reshape(gt.shape[0],gt.shape[1]),p=2)).item(),
        # 's_SSIM':ss.item(),
        'SSIM': s,
        "NRMSE": n
        })
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
            
            
    df = pd.DataFrame(record)
    df.to_excel('result/figure/seed%d_loss data.xlsx'%(seed), index=False)