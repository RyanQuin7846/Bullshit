# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 18:22:07 2024

@author: Server2
"""
import numpy as np
from scipy.sparse import csc_matrix

def sparse_puxy_mix(mat_Lx, mat_Ly, mat_Lx_g, mat_Ly_g, mat_d1xA, mat_d1yA, dx, dy, Nx, Ny):
    nx = Nx-2
    ny = Ny-2
    cv_Lx = np.reshape(mat_Lx, (nx*ny), order='F')
    cv_Ly = np.reshape(mat_Ly, (nx*ny), order='F')
    cv_Lx_g = np.reshape(mat_Lx_g, (nx*ny), order='F')
    cv_Ly_g = np.reshape(mat_Ly_g, (nx*ny), order='F')
    idx_row = np.zeros((3*nx*ny))
    idx_col = np.zeros((3*nx*ny))
    idx_d1x = np.empty((3*nx*ny))
    idx_d1xA = np.empty((3*nx*ny))#,dtype=complex)
    idx_d1xAG = np.empty((3*nx*ny))#,dtype=complex)
    idy_row = np.zeros((3*nx*ny))
    idy_col = np.zeros((3*nx*ny))
    idy_d1y = np.empty((3*nx*ny))
    idy_d1yA = np.empty((3*nx*ny))#,dtype=complex)
    idy_d1yAG = np.empty((3*nx*ny))#,dtype=complex)
    rpos = -1 #-1
    cpos = Nx-1 #Ny-1
    counter = 0
    for yy in range (ny):
        for xx in range (nx):
            ipos = (yy)*nx+xx
            rpos = rpos+1
            cpos = cpos+1
            idx_row[counter:counter+3]= np.asarray([rpos,rpos,rpos])
            idx_col[counter:counter+3]= np.asarray([cpos,cpos+1,cpos+2])
            idx_d1x[counter:counter+3] = np.asarray([1,0,1])
            idx_d1xA[counter:counter+3]= np.asarray([cv_Lx[ipos],cv_Lx[ipos],cv_Lx[ipos]])
            idx_d1xAG[counter:counter+3]= np.asarray([cv_Lx_g[ipos],cv_Lx_g[ipos],cv_Lx_g[ipos]])
            px = xx+1
            py = yy+1 #yy+2
            fp = (py+1)*Nx+px
            cp = py*Nx+px
            lp = (py-1)*Nx+px
            idy_row[counter:counter+3]= np.asarray([rpos,rpos,rpos])
            idy_col[counter:counter+3]= np.asarray([fp,cp,lp])
            idy_d1yA[counter:counter+3]= np.asarray([cv_Ly[ipos],cv_Ly[ipos],cv_Ly[ipos]])
            idy_d1yAG[counter:counter+3]= np.asarray([cv_Ly_g[ipos],cv_Ly_g[ipos],cv_Ly_g[ipos]])
            idy_d1y[counter:counter+3] = np.asarray([1,0,1])
            counter = counter+3
            if ((rpos+1)%nx == 0):#if (rpos!= 0) and (rpos%nx == 0):
                cpos = cpos+2
    spmat_Lx = csc_matrix((idx_d1xA, (idx_row, idx_col)), shape=(nx*ny, (Nx)*(Ny))) 
    spmat_Ly = csc_matrix((idy_d1yA, (idy_row, idy_col)), shape=(nx*ny, (Nx)*(Ny)))
    spmat_x = csc_matrix((idx_d1x, (idy_row, idy_col)), shape=(nx*ny, (Nx)*(Ny)))
    spmat_y = csc_matrix((idy_d1y, (idy_row, idy_col)), shape=(nx*ny, (Nx)*(Ny)))
    spmat_Lx_g = csc_matrix((idx_d1xAG, (idx_row, idx_col)), shape=(nx*ny, (Nx)*(Ny))) 
    spmat_Ly_g = csc_matrix((idy_d1yAG, (idy_row, idy_col)), shape=(nx*ny, (Nx)*(Ny)))
    mat_pux = spmat_Lx.multiply(spmat_Lx_g).multiply(spmat_x)#/(2*dx) 
    mat_puy = spmat_Ly.multiply(spmat_Ly_g).multiply(spmat_y)#/(2*dy)
    return mat_pux, mat_puy, spmat_Lx, spmat_Ly, spmat_x, spmat_y, spmat_Lx_g, spmat_Ly_g