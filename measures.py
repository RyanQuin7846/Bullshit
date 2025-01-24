# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 18:26:28 2024

@author: Server2
"""

import numpy as np
from skimage.metrics import structural_similarity as ssiml

def measures(im1, im2):
    im1 = im1.astype(np.float64)
    im2 = im2.astype(np.float64)
    msex = np.mean((np.subtract(im1,im2))**2)
    maxx = im1.max()
    nrmsex = (np.sqrt(msex)/(im1.max()-im1.min()))
    psnrx = 20*np.log10(np.abs(maxx)/np.sqrt(msex))
    ssimx = ssiml(im1,im2)
    return msex, nrmsex, psnrx, ssimx