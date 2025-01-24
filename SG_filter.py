# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 18:25:23 2024

@author: Server2
"""

import numpy as np

cv_unit = np.ones((9,1))

def SG_filter(cH1p, matx, maty):
    Nx, Ny = cH1p.shape
    nx, ny = Nx-2, Ny-2
    mat_d1xH1p = np.empty((nx,ny))
    mat_d1yH1p = np.empty((nx,ny))
    mat_lapH1p = np.empty((nx,ny))
    for yy in range (0,ny):
        for xx in range (0,nx):  
            rpos = xx+1
            cpos = yy+1
            np_H1p = cH1p[rpos-1:rpos+2, cpos-1:cpos+2]
            cvnp_H1p = np.reshape(np_H1p,(9,1), order='F')
            np_x = matx[rpos-1:rpos+2, cpos-1:cpos+2]
            cvnp_x = np.reshape(np_x,(9,1), order='F')
            np_y = maty[rpos-1:rpos+2, cpos-1:cpos+2]
            cvnp_y = np.reshape(np_y,(9,1), order='F')
            cx = np_x[1,1]
            cy = np_y[1,1]
            mat_A = np.concatenate((cvnp_x**2,cvnp_y**2,cvnp_x*cvnp_y,cvnp_x,cvnp_y,cv_unit),axis=1)
            cv_a = np.linalg.solve(np.matmul(np.transpose(mat_A),mat_A),np.matmul(np.transpose(mat_A),cvnp_H1p))
            mat_d1xH1p[xx,yy] = 2*cv_a[0]*cx + cv_a[2]*cy + cv_a[3]
            mat_d1yH1p[xx,yy] = 2*cv_a[1]*cy + cv_a[2]*cx + cv_a[4]
            mat_lapH1p[xx,yy] = 2*cv_a[0] + 2*cv_a[1]
    mat_lap = (mat_lapH1p)
    mat_Lx = (mat_d1xH1p) 
    mat_Ly = (mat_d1yH1p) 
    return mat_lap, mat_Lx, mat_Ly