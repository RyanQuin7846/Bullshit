# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 15:39:25 2024

@author: Server2
"""

import numpy as np
import matplotlib.pyplot as plt

condy = np.load('data/raw data/no tumor/condy.npy')

for i in range(condy.shape[0]):
        plt.clf()
        plt.imshow(condy[i],clim=(0,3),cmap='coolwarm')
        plt.xticks([])
        plt.yticks([])
        plt.colorbar()
        plt.savefig("data/raw data/no tumor/slice%s.jpg"%(i),bbox_inches='tight')