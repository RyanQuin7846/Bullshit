# -*- coding: utf-8 -*-
"""
Created on Mon Aug 19 09:25:47 2024

@author: Server2
"""
import numpy as np
import math
from scipy.sparse import csc_matrix, find, linalg

from sparse_matrices import sparse_matrices
from SG_filter import SG_filter
from sparse_puxy import sparse_puxy

def stab_ept(ogmatx,ogmaty,phH1p,ogcond,delta):
    Nx = phH1p.shape[1]
    Ny = phH1p.shape[0]
    nx, ny = Nx-2, Ny-2
    
    nx, ny = Nx-2, Ny-2 
    u0 = 4*math.pi*1e-7
    freq = 127.8e6
    omega = 2*math.pi*freq 
    cH1p = phH1p.copy()
    matx = ogmatx.copy()
    maty = ogmaty.copy()
    dy = np.array((np.max(ogmaty)-np.min(ogmaty))/(Ny-1))
    dx = np.array((np.max(ogmatx)-np.min(ogmatx))/(Ny-1))
    Nx = cH1p.shape[0]
    Ny = cH1p.shape[1]
    nx = Nx-2
    ny = Ny-2    
    
    
    mat_d1xA, mat_d2xA, mat_d1yA, mat_d2yA, ind1xa, ind1ya, stencil_x, stencil_y = sparse_matrices(Nx,Ny)
    mat_d1xAl2, mat_d2xAl2, mat_d1yAl2, mat_d2yAl2, ind1xal2, ind1yal2, _, _ = sparse_matrices(nx,ny)
    
    mat_lap, mat_Lx, mat_Ly = SG_filter(cH1p,matx,maty)
    std_cond = mat_lap/(omega*u0)
    std_lap, std_Lx, std_Ly = SG_filter(1/std_cond,matx[1:-1,1:-1],maty[1:-1,1:-1])
    mat_pux,mat_puy,spmat_Lx,spmat_Ly = sparse_puxy(mat_Lx, mat_Ly, mat_d1xA, mat_d1yA, dx, dy, Nx, Ny)
    puro = (mat_pux + mat_puy)
    deltax = delta
    mat_ppu = np.multiply(deltax,mat_d2xA.todense())/(dx**2) + np.multiply(deltax,mat_d2yA.todense())/(dy**2)
    mat_A = puro - mat_ppu
    cv_lap = np.reshape(mat_lap, (nx*ny), order='F')
    pos_u = np.empty((nx*ny),)
    for pp in range (0,nx*ny):
        pos_u[pp] = pp
    mat_bnd = csc_matrix((Nx,Ny))
    mat_bnd[0,:] = ogcond[0,:]
    mat_bnd[:,0] = ogcond[:,0]
    mat_bnd[-1,:] = ogcond[-1,:]  
    mat_bnd[:,-1] = ogcond[:,-1]  
    cv_bnd = np.reshape(mat_bnd,(Nx*Ny,1), order='F')
    all_bnd = find(cv_bnd)
    to_keep = list(set(range(Nx*Ny))-set(all_bnd[0]))  
    to_keep = np.asarray(to_keep)
    cv_b = (omega*u0)*np.ones((nx*ny,1))
    mat_u = csc_matrix((cv_lap, (pos_u, pos_u)), shape=(nx*ny, nx*ny)).todense() 
    mat_A2 = mat_A[:,to_keep]
    mat_Ax = mat_A2 + mat_u
    gamma = linalg.spsolve(mat_Ax,cv_b)
    stab_cond = np.reshape((1/gamma),(nx,ny), order='F')
    
    return stab_cond