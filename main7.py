# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 16:33:21 2024

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
from data_maker import data_maker
from normalize_matrix import normalize_matrix
from torch_normalize_matrix import torch_normalize_matrix
from stabEPT import stab_ept
from set_seed import set_seed
from Dnetworks import FCNN
from SG_filter import SG_filter

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
    gradient_y = gradient_y[1:-1,1:-1]
    gradient_x = gradient_x[1:-1,1:-1]
    laplacian = laplacian[1:-1,1:-1]
    stab = stab[1:-1,1:-1]
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

omega = 2 * math.pi * 127.8e6
n_best = 100
s_best = -100

# mat_lap, mat_Lx, mat_Ly = SG_filter(phase, y, x)
# laplacian = mat_lap
# gradient = mat_Lx + mat_Ly

# stab = stab_ept(y, x, phase, gt, 0.01)

x = x[1:-1, 1:-1]
y = y[1:-1, 1:-1]
phase = phase[1:-1, 1:-1]
gt = gt[1:-1, 1:-1]
# pd_s = pd_s[1:-1, 1:-1]

plt.clf()
plt.imshow(gt, clim=(0, 3))
plt.xticks([])
plt.yticks([])
plt.colorbar()
plt.savefig("result/figure/gt.jpg", bbox_inches='tight')

# 创建三个不同的模型实例
net_std = FCNN()
net_cr = FCNN()
net_stab = FCNN()

# 优化器
optimizer_std = torch.optim.Adam(net_std.parameters(), lr=5e-4, betas=(0.5, 0.999))
optimizer_cr = torch.optim.Adam(net_cr.parameters(), lr=5e-4, betas=(0.5, 0.999))
optimizer_stab = torch.optim.Adam(net_stab.parameters(), lr=5e-4, betas=(0.5, 0.999))

# 损失函数
mse_loss = nn.MSELoss()

cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

if cuda:
    net_std.cuda()
    net_cr.cuda()
    net_stab.cuda()
    mse_loss.cuda()

# 数据转换
phase = torch.tensor(phase.reshape(-1, 1), requires_grad=True).type(Tensor)
gradient_x = torch.tensor(gradient_x.reshape(-1, 1), requires_grad=True).type(Tensor)
gradient_y = torch.tensor(gradient_y.reshape(-1, 1), requires_grad=True).type(Tensor)
laplacian = torch.tensor(laplacian.reshape(-1, 1), requires_grad=True).type(Tensor)
gt_t = torch.tensor(gt, requires_grad=True).type(Tensor)
x = torch.tensor(x.reshape(-1, 1), requires_grad=True).type(Tensor)
y = torch.tensor(y.reshape(-1, 1), requires_grad=True).type(Tensor)

epochs = 500000

# 定义用于保存和处理图像的函数
def save_image(e, epoch, n, s, seed, loss_type,loss):
    plt.clf()
    plt.imshow(e)
    plt.title(f"{loss_type}\nEPOCH={epoch + 1}\nNRMSE={n:.3f}\nSSIM={s:.3f}\nLoss={loss:.3f}", fontproperties='Times New Roman', fontsize=25)
    plt.xticks([])
    plt.yticks([])
    plt.colorbar()
    plt.savefig(f"E:/Qin/PINN/result/figure/seed{seed}_Best_{loss_type}.jpg", bbox_inches='tight')

# 训练网络的过程，分别为三个模型训练
for seed in range(4399): 
    set_seed(seed)

    for epoch in range(epochs):
        print(epoch)
        
        # Std Loss 网络
        sigma_std = net_std(x, y, phase, gradient_x, gradient_y, laplacian)
        gamma_std = 1/sigma_std.reshape(gt.shape[0], gt.shape[1])
        mse1 = mse_loss(sigma_std, torch.tensor(stab).type(Tensor).reshape(-1,1))
        std_loss = ((gamma_std * laplacian.reshape(gt.shape[0], gt.shape[1]))
                    - 2 * (2 * math.pi * 127.8e6 * 4e-7 * math.pi))
        loss_std = mse1*1e3 + torch.norm(std_loss, p=2)
        optimizer_std.zero_grad()
        loss_std.backward()
        optimizer_std.step()

        sigma_std = sigma_std.detach().cpu().numpy().reshape(gt.shape[0], gt.shape[1])
        n_std = nrmse(sigma_std, gt)
        s_std = ssim(sigma_std, gt, data_range=gt.max() - gt.min())

        if epoch % 10 == 0:
            save_image(sigma_std, epoch, n_std, s_std, seed, 'Std',loss_std.detach().cpu().numpy())

        # CR Loss 网络
        sigma_cr = net_cr(x, y, phase, gradient_x, gradient_y, laplacian)
        gamma_cr = 1/sigma_cr.reshape(gt.shape[0], gt.shape[1])
        p_g_x_cr = torch.autograd.grad(gamma_cr.sum(), x, create_graph=True, retain_graph=True)[0].reshape(gt.shape[0], gt.shape[1])
        p_g_y_cr = torch.autograd.grad(gamma_cr.sum(), y, create_graph=True, retain_graph=True)[0].reshape(gt.shape[0], gt.shape[1])
        mse2 = mse_loss(sigma_cr, torch.tensor(stab).type(Tensor).reshape(-1,1))
        cr_loss = ((p_g_x_cr * gradient_x.reshape(gt.shape[0], gt.shape[1])+p_g_y_cr * gradient_y.reshape(gt.shape[0], gt.shape[1]))
                   + (gamma_cr * laplacian.reshape(gt.shape[0], gt.shape[1]))
                   - 2 * (2 * math.pi * 127.8e6 * 4e-7 * math.pi))
        loss_cr = mse2*1e3 + torch.norm(cr_loss, p=2)
        optimizer_cr.zero_grad()
        loss_cr.backward()
        optimizer_cr.step()

        sigma_cr = sigma_cr.detach().cpu().numpy().reshape(gt.shape[0], gt.shape[1])
        n_cr = nrmse(sigma_cr, gt)
        s_cr = ssim(sigma_cr, gt, data_range=gt.max() - gt.min())

        if epoch % 10 == 0:
            save_image(sigma_cr, epoch, n_cr, s_cr, seed, 'CR',loss_cr.detach().cpu().numpy())

        # Stab Loss 网络
        sigma_stab = net_stab(x, y, phase, gradient_x, gradient_y, laplacian)
        gamma_stab = 1/sigma_stab.reshape(gt.shape[0], gt.shape[1])
        p_g_x_stab = torch.autograd.grad(gamma_stab.sum(), x, create_graph=True, retain_graph=True)[0].reshape(gt.shape[0], gt.shape[1])
        p_g_y_stab = torch.autograd.grad(gamma_stab.sum(), y, create_graph=True, retain_graph=True)[0].reshape(gt.shape[0], gt.shape[1])
        p2_g_x_stab = torch.autograd.grad(p_g_x_stab.sum(), x, create_graph=True, retain_graph=True)[0].reshape(gt.shape[0], gt.shape[1])
        p2_g_y_stab = torch.autograd.grad(p_g_y_stab.sum(), y, create_graph=True, retain_graph=True)[0].reshape(gt.shape[0], gt.shape[1])
        p2_g =  p2_g_x_stab + p2_g_y_stab
        mse3 = mse_loss(sigma_stab, torch.tensor(stab).type(Tensor).reshape(-1,1))
        stab_loss = (-0.01 * p2_g.reshape(gt.shape[0], gt.shape[1]) + (p_g_x_stab * gradient_x.reshape(gt.shape[0], gt.shape[1])+p_g_y_stab * gradient_y.reshape(gt.shape[0], gt.shape[1]))
                     + (gamma_stab * laplacian.reshape(gt.shape[0], gt.shape[1]))
                     - 2 * (2 * math.pi * 127.8e6 * 4e-7 * math.pi))
        loss_stab = mse3*1e3 + torch.norm(stab_loss, p=2)
        optimizer_stab.zero_grad()
        loss_stab.backward()
        optimizer_stab.step()

        sigma_stab = sigma_stab.detach().cpu().numpy().reshape(gt.shape[0], gt.shape[1])
        n_stab = nrmse(sigma_stab, gt)
        s_stab = ssim(sigma_stab, gt, data_range=gt.max() - gt.min())

        if epoch % 10 == 0:
            save_image(sigma_stab, epoch, n_stab, s_stab, seed, 'Stab',loss_stab.detach().cpu().numpy())
