# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 18:24:26 2024

@author: Server2
"""

import numpy as np
from scipy.sparse import csc_matrix

def sparse_matrices(Nx,Ny):
    nx, ny = Nx-2, Ny-2
    idx_row = np.zeros((3*nx*ny))
    idy_row = np.zeros((3*nx*ny))
    idx_col = np.zeros((3*nx*ny))
    idy_col = np.zeros((3*nx*ny))
    id_d1x = np.zeros((3*nx*ny))
    id_d2x = np.zeros((3*nx*ny))
    id_d1y = np.zeros((3*nx*ny))
    id_d2y = np.zeros((3*nx*ny))
    id_d1xx = np.zeros((3*nx*ny))
    id_d1yy = np.zeros((3*nx*ny))
    rpos = -1
    cpos = Nx-1
    counter = 0 
    for yy in range (ny):
        for xx in range (nx):
            rpos = rpos+1 
            cpos = cpos+1
            idx_row[counter:counter+3] = np.asarray([rpos,rpos,rpos])
            idx_col[counter:counter+3] = np.asarray([cpos,cpos+1,cpos+2])
            id_d1x[counter:counter+3] = np.asarray([-1,0,1])
            id_d1xx[counter:counter+3] = np.asarray([1,0,1])
            id_d2x[counter:counter+3] = np.asarray([1,-2,1])
            px = xx+1
            py = yy+1
            fp = (py+1)*Nx+px
            cp = py*Nx+px
            lp = (py-1)*Nx+px
            idy_row[counter:counter+3] = np.asarray([rpos,rpos,rpos])
            idy_col[counter:counter+3] = np.asarray([fp,cp,lp])
            id_d1y[counter:counter+3] = np.asarray([1,0,-1])
            id_d1yy[counter:counter+3] = np.asarray([1,0,1])
            id_d2y[counter:counter+3] = np.asarray([1,-2,1])
            counter = counter+3
            if ((rpos+1)%nx == 0):
                cpos = cpos+2
    mat_d1xA = csc_matrix((id_d1x, (idx_row, idx_col)), shape=(nx*ny, (Nx)*(Ny))) 
    mat_d2xA = csc_matrix((id_d2x, (idx_row, idx_col)), shape=(nx*ny, (Nx)*(Ny)))
    mat_d1yA = csc_matrix((id_d1y, (idy_row, idy_col)), shape=(nx*ny, (Nx)*(Ny))) 
    mat_d2yA = csc_matrix((id_d2y, (idy_row, idy_col)), shape=(nx*ny, (Nx)*(Ny)))
    mat_d1xxA = csc_matrix((id_d1x, (idx_row, idx_col)), shape=(nx*ny, (Nx)*(Ny))) 
    mat_d1yyA = csc_matrix((id_d1y, (idy_row, idy_col)), shape=(nx*ny, (Nx)*(Ny))) 
    ind1xa = np.vstack((idx_row, idx_col))
    ind1ya = np.vstack((idy_row, idy_col))
    return mat_d1xA, mat_d2xA, mat_d1yA, mat_d2yA, ind1xa, ind1ya, mat_d1xxA, mat_d1yyA