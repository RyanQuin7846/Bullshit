# -*- coding: utf-8 -*-
"""
Created on Fri Nov 29 16:47:11 2024

@author: Server2
"""

import numpy as np
import matplotlib.pyplot as plt

# 加载数据
data = np.load("E:/Qin/PINN/result/figure/20250102 1e5+collocation points/data_epoch1e5_plus_minus_100.npz")

# 提取数据
s = data['s']
s_std = data['s_std']
s_cr = data['s_cr']
s_stab = data['s_stab']

# 创建 x 轴
x = np.arange(len(s))  # 假设 s, s_std, s_cr, s_stab 长度相同

# 绘制曲线
plt.figure(figsize=(10, 6))  # 设置画布大小
plt.plot(x, s, label='Ground truth', linewidth=1.5)
plt.plot(x, s_std, label='Std-EPT', linewidth=1.5)
plt.plot(x, s_cr, label='CR-EPT', linewidth=1.5)
plt.plot(x, s_stab, label='Stab-EPT', linewidth=1.5)
# plt.xscale('log')

# 设置 y 轴范围
plt.ylim(0, 1)

# 添加标题和标签
plt.title('Comparison of SSIM Curves', fontproperties='Times New Roman', fontsize=20)
# plt.xlabel('Step', fontproperties='Times New Roman', fontsize=15)
# plt.ylabel('Value', fontproperties='Times New Roman', fontsize=15)

# 添加图例
plt.legend(fontsize=12)

# 显示网格
plt.grid(True, linestyle='--', alpha=0.5)

# 显示图像
plt.tight_layout()
plt.show()
