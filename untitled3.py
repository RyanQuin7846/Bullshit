# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 17:08:01 2024

@author: Server2
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams['font.family'] = 'Times New Roman'

data = np.load("E:/Qin/PINN/result/figure/20250102 1e5+collocation points/data_epoch1e5_plus_minus_100.npz")

loss = data['loss']
s = data['s']
s_std = data['s_std']
s_cr = data['s_cr']
s_stab = data['s_stab']
# p_g_p_x = data['p_g_p_x'][-1]
# # p_g_p_x = np.sqrt(np.sum(p_g_p_x**2, axis=(1, 2)))
# p_g_p_y = data['p_g_p_y'][-1]
# # p_g_p_y = np.sqrt(np.sum(p_g_p_y**2, axis=(1, 2)))
# p2_g = data['p2_g']
# p2_g = np.sqrt(np.sum(p2_g**2, axis=(1, 2)))

# plt.clf()
# plt.imshow(p_g_p_x)
# # plt.title("NRMSE=%1.3f\nSSIM=%1.3f"%(nrmse(stab,gt),ssim(stab,gt,data_range=gt.max()-gt.min())),fontproperties = 'Times New Roman',fontsize = 25)
# plt.xticks([])
# plt.yticks([])
# plt.colorbar()
# plt.show()

# plt.clf()
# plt.imshow(p_g_p_y)
# # plt.title("NRMSE=%1.3f\nSSIM=%1.3f"%(nrmse(stab,gt),ssim(stab,gt,data_range=gt.max()-gt.min())),fontproperties = 'Times New Roman',fontsize = 25)
# plt.xticks([])
# plt.yticks([])
# plt.colorbar()
# plt.show()
x1 = np.arange(1,1e5+1)
plt.clf()
plt.plot(x1,loss[0:100000])
# plt.xscale('log')
plt.title('MSE loss step',fontproperties = 'Times New Roman',fontsize = 25)
# plt.xticks([])
# plt.yticks([])
# plt.colorbar()
plt.show()

x2 = np.arange(1e5+1,5e5-1)
plt.clf()
plt.plot(x2, loss[100001:-1])
# plt.xscale('log')
plt.title( 'PDE loss step',fontproperties = 'Times New Roman',fontsize = 25)
# plt.xticks([100001,500000])
# plt.yticks([])
# plt.colorbar()
plt.show()

y = (x2-1e5)/(4e5)
# plt.clf()
# plt.plot(x2,y)
# plt.title('Anealing coefficient',fontproperties = 'Times New Roman',fontsize = 25)
# # plt.xticks([])
# # plt.yticks([])
# # plt.colorbar()
# plt.show()

# fig, ax1 = plt.subplots()

# # Plot the loss on the left y-axis
# ax1.plot(x2, loss[100001:-1], 'b-', label='PDE loss step')
# ax1.set_xlabel('Step')
# ax1.set_ylabel('PDE loss', color='b', fontproperties='Times New Roman', fontsize=15)
# ax1.tick_params(axis='y', labelcolor='b')

# # Create a second y-axis for the annealing coefficient
# ax2 = ax1.twinx()
# ax2.plot(x2, y, 'r-', label='Annealing coefficient')
# ax2.set_ylabel('Annealing coefficient', color='r', fontproperties='Times New Roman', fontsize=15)
# ax2.tick_params(axis='y', labelcolor='r')

# plt.title('PDE loss step and Annealing coefficient', fontproperties='Times New Roman', fontsize=25)
# fig.tight_layout()
# plt.show()
plt.clf()
plt.plot(s)
plt.title('SSIM',fontproperties = 'Times New Roman',fontsize = 25)
# plt.xticks([])
# plt.yticks([])
# plt.colorbar()
plt.show()

# plt.clf()
# plt.plot(p2_g)
# plt.ylim([0,5e6])
# # plt.xticks([])
# # plt.yticks([])
# # plt.colorbar()
# plt.show()