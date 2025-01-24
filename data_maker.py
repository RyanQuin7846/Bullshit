# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 18:32:48 2024

@author: Server2
"""

import os
import numpy as np
import math
from scipy.sparse import csc_matrix, find, linalg

from sparse_puxy_mix import sparse_puxy_mix
from loader_gen import load_gen
from sparse_matrices import sparse_matrices
from SG_filter import SG_filter
from sparse_puxy import sparse_puxy
from measures import measures

def data_maker(delta,Flag='all'):
    struct_id = "Tumor_sizes"
    tt = os.listdir('Tumor_sizes')
    mx = {}
    my = {}
    mz = {}
    pH1p = {}
    aH1p = {} 
    con = {}
    per = {}
    for x in range (len(tt)):
        mx[x], my[x], mz[x], pH1p[x], aH1p[x], con[x], per[x] = load_gen(struct_id,str(tt[x]))
    Nx = 38#slsimp 32# phH1p.shape[2]
    Ny = 38#slsimp 32#phH1p.shape[3]
    nx, ny = Nx-2, Ny-2
    idxn = 1
    ogmatx = np.concatenate([mx[x] for x in sorted(mx)], 0)[idxn,:,:,0]
    ogmaty = np.concatenate([my[x] for x in sorted(my)], 0)[idxn,:,:,0]
    phH1p = np.concatenate([pH1p[x] for x in sorted(pH1p)], 0)[idxn,0]
    ogcond = np.concatenate([con[x] for x in sorted(con)], 0)[idxn,0]
    
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
    mat_bnd[0,:] = 1
    mat_bnd[:,0] = 1 
    mat_bnd[-1,:] = 1  
    mat_bnd[:,-1] = 1  
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
    if Flag == 'stab':
        return stab_cond
    else:
        stab_lap, stab_Lx, stab_Ly = SG_filter(1/stab_cond,matx[1:-1,1:-1],maty[1:-1,1:-1])
        
        gamma_lap, gamma_Lx, gamma_Ly = SG_filter(1/stab_cond,matx[1:-1,1:-1],maty[1:-1,1:-1])
        mse_std_, nrmse_std_, psnr_std_, ssim_std_ = measures(ogcond[1:-1,1:-1],std_cond)
        mse_stab_, nrmse_stab_, psnr_stab_, ssim_stab_ = measures(ogcond[1:-1,1:-1],stab_cond)
        
        mx, my = nx-2, ny-2
        cv_lap = np.reshape(mat_lap[1:-1,1:-1], (mx*my), order='F')
        pos_u = np.empty((mx*my),)
        for pp in range (0,mx*my):
            pos_u[pp] = pp
        mat_bnd = csc_matrix((Nx,Ny))
        mat_bnd[0,:] = 1
        mat_bnd[1,:] = 1
        mat_bnd[:,0] = 1 
        mat_bnd[:,1] = 1 
        mat_bnd[-1,:] = 1  
        mat_bnd[:,-1] = 1  
        mat_bnd[-2,:] = 1  
        mat_bnd[:,-2] = 1  
        cv_bnd = np.reshape(mat_bnd,(Nx*Ny,1), order='F')
        all_bnd = find(cv_bnd)
        to_keep = list(set(range(Nx*Ny))-set(all_bnd[0]))  
        to_keep = np.asarray(to_keep)
        
        mat_bnd2 = csc_matrix((nx,ny))
        mat_bnd2[0,:] = 1
        mat_bnd2[:,0] = 1 
        mat_bnd2[-1,:] = 1  
        mat_bnd2[:,-1] = 1  
        cv_bnd2 = np.reshape(mat_bnd2,(nx*ny,1), order='F')
        all_bnd2 = find(cv_bnd2)
        to_keep2 = list(set(range(nx*ny))-set(all_bnd2[0]))  
        to_keep2 = np.asarray(to_keep2)
        
        mat_pux2, mat_puy2, spmat_Lx2, spmat_Ly2, sten_x, sten_y, spmat_Lx_g, spmat_Ly_g = sparse_puxy_mix(mat_Lx[1:-1,1:-1], mat_Ly[1:-1,1:-1], 
                                                                   gamma_Lx, gamma_Ly, mat_d1xAl2, 
                                                                   mat_d1yAl2, dx, dy, nx, ny)
        deltaxx = 0.005
        puro = mat_pux2 + mat_puy2
        mat_ppu = np.multiply(deltaxx,mat_d2xAl2.todense())/(dx**2) + np.multiply(deltaxx,mat_d2yAl2.todense())/(dy**2)
        
        return matx, maty, cH1p, mat_Lx, mat_Ly, mat_lap, stab_cond, ogcond