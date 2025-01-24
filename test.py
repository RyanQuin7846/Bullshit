# -*- coding: utf-8 -*-
"""
Created on Mon Aug 19 09:32:24 2024

@author: Server2
"""

import numpy as np

from stabEPT import stab_ept

x = np.load("data/raw data/4mm tumor double/X.npy")[0].T
y = np.load("data/raw data/4mm tumor double/Y.npy")[0].T
phase = np.load("data/raw data/4mm tumor double/H.npy")[0].imag.T
gt = np.load("data/raw data/4mm tumor double/condy.npy")[0].T

stab = stab_ept(x, y, phase, gt, 0.02)