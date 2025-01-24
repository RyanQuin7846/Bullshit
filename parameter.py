# -*- coding: utf-8 -*-
"""
Created on Mon Sep  2 14:15:03 2024

@author: Server2
"""

import torch
import torch.nn as nn
import numpy as np
import math
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from SG_filter import SG_filter

# 定义线性模型
class LinearModel(nn.Module):
    def __init__(self):
        super(LinearModel, self).__init__()
        self.linear = nn.Linear(3, 1, bias=False)  # 3 inputs (x, y, z) and 1 output (ax + by + cz)
        # 手动初始化权重为0
        nn.init.constant_(self.linear.weight, 0.0)
    def forward(self, x):
        return self.linear(x)

# 创建模型实例
model = LinearModel()

# 损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.00001)

# 输入数据 (x, y, z)
matrix_x = np.load("data/raw data/4mm tumor pos1/X.npy")
x = matrix_x[1].T
matrix_y = np.load("data/raw data/4mm tumor pos1/Y.npy")
y = matrix_y[1].T
phase = np.load("data/raw data/4mm tumor pos1/H.npy")
# phase = phase[1].imag.T
phH1p = np.zeros((1,phase.shape[0],phase.shape[1],phase.shape[2]))
for mm in range(2):
    phH1p[:,mm,:] = np.unwrap(np.angle(phase[mm,:,:],deg=False))
phase = phH1p[0,0].T
gt = np.load("data/raw data/4mm tumor pos1/condy.npy")
gt = gt[1].T
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

cuda = True if torch.cuda.is_available() else False

if cuda:
    model.cuda()
    criterion.cuda()

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
phase = torch.tensor(phase.reshape(-1,1) ).type(Tensor)
gradient = torch.tensor(gradient.reshape(-1,1) ).type(Tensor)
laplacian = torch.tensor(laplacian.reshape(-1,1) ).type(Tensor)
gamma = torch.tensor(gamma.reshape(-1,1) ).type(Tensor)
c_gradient = torch.tensor(c_gradient.reshape(-1,1) ).type(Tensor)
c_laplacian = torch.tensor(c_laplacian.reshape(-1,1) ).type(Tensor)

# 定义函数用于计算均值、标准差，并移除离群值
def remove_outliers(data_tensor, name):
    mean = data_tensor.mean().item()  # 计算均值
    std = data_tensor.std().item()    # 计算标准差
    lower_bound = mean - 1 * std      # 下限
    upper_bound = mean + 1 * std      # 上限
    
    print(f"{name} - Mean: {mean}, Std: {std}, Lower bound: {lower_bound}, Upper bound: {upper_bound}")
    
    # 保留范围内的值
    mask = (data_tensor >= lower_bound) & (data_tensor <= upper_bound)
    
    return mask

# 计算并移除离群值
phase_mask = remove_outliers(phase, 'Phase')
gradient_mask = remove_outliers(gradient, 'Gradient')
laplacian_mask = remove_outliers(laplacian, 'Laplacian')
gamma_mask = remove_outliers(gamma, 'Gamma')
c_gradient_mask = remove_outliers(c_gradient, 'C_Gradient')
c_laplacian_mask = remove_outliers(c_laplacian, 'C_Laplacian')

# 仅保留所有掩码都为True的位置
combined_mask = phase_mask & gradient_mask & laplacian_mask & gamma_mask & c_gradient_mask & c_laplacian_mask

# 使用相同的掩码过滤输入数据
phase = phase[combined_mask].reshape(-1,1)
gradient = gradient[combined_mask].reshape(-1,1)
laplacian = laplacian[combined_mask].reshape(-1,1)
gamma = gamma[combined_mask].reshape(-1,1)
c_gradient = c_gradient[combined_mask].reshape(-1,1)
c_laplacian = c_laplacian[combined_mask].reshape(-1,1)


inputs = torch.cat((c_laplacian,(gradient*c_gradient),(gamma*laplacian)), axis=1)/1e3

# 目标输出 (目标是 ax + by + cz = 0, 所以目标是 0)
targets = torch.zeros(inputs.size(0), 1).type(Tensor)  # 全部为0
targets[:] = (math.pi*(127.8e6)*(4e-7)*math.pi)/1e3 

# 创建 DataLoader
dataset = TensorDataset(inputs, targets)
batch_size = 512  # 设置批大小
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# 训练
for epoch in range(1000000):
    for batch_inputs, batch_targets in dataloader:
        optimizer.zero_grad()
        outputs = model(batch_inputs)
        loss = criterion(outputs, batch_targets)
        print(f'Epoch [{epoch + 1}/1000], Loss: {loss.item():.4f}')
        loss.backward()
        optimizer.step()
    
    # if (epoch + 1) % 100 == 0:
    # print(f'Epoch [{epoch + 1}/1000], Loss: {loss.item():.4f}')

    # 打印学习到的系数 a, b, c
    learned_coefficients = model.linear.weight.data
    print(f'Learned coefficients: a={learned_coefficients[0, 0]}, b={learned_coefficients[0, 1]}, c={learned_coefficients[0, 2]}')
