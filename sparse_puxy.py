# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 11:22:00 2019

@author: yulab
"""
import numpy as np
from scipy.sparse import csc_matrix
def sparse_puxy(mat_Lx, mat_Ly, mat_d1xA, mat_d1yA, dx, dy, Nx, Ny):
    nx = Nx-2
    ny = Ny-2
    cv_Lx = np.reshape(mat_Lx, (nx*ny), order='F')
    cv_Ly = np.reshape(mat_Ly, (nx*ny), order='F')
#    idx_row = np.zeros((9*nx*ny))
#    idx_col = np.zeros((9*nx*ny))
#    idx_d1xA = np.empty((9*nx*ny),dtype=complex)
#    idy_row = np.zeros((9*nx*ny))
#    idy_col = np.zeros((9*nx*ny))
#    idy_d1yA = np.empty((9*nx*ny),dtype=complex)
    idx_row = np.zeros((3*nx*ny))
    idx_col = np.zeros((3*nx*ny))
    idx_d1xA = np.empty((3*nx*ny))#,dtype=complex)
    idy_row = np.zeros((3*nx*ny))
    idy_col = np.zeros((3*nx*ny))
    idy_d1yA = np.empty((3*nx*ny))#,dtype=complex)
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
            idx_d1xA[counter:counter+3]= np.asarray([cv_Lx[ipos],cv_Lx[ipos],cv_Lx[ipos]])
#            idx_row[counter:counter+9] = np.asarray([rpos,rpos,rpos,rpos,rpos,rpos,rpos,rpos,rpos])   
#            idx_col[counter:counter+9] = np.asarray([cpos-Ny,cpos+1-Ny,cpos+2-Ny,cpos,cpos+1,cpos+2,cpos+(Ny),cpos+1+(Ny),cpos+2+(Ny)])
#            idx_d1xA[counter:counter+9]= np.asarray([cv_Lx[ipos],cv_Lx[ipos],cv_Lx[ipos],cv_Lx[ipos],cv_Lx[ipos],cv_Lx[ipos],cv_Lx[ipos],cv_Lx[ipos],cv_Lx[ipos]])
#            idx_row[counter:counter+5] = np.asarray([rpos,rpos,rpos,rpos,rpos])   
#            idx_col[counter:counter+5] = np.asarray([cpos,cpos+1,cpos+2,cpos+3,cpos+4])
#            idx_d1xA[counter:counter+5]= np.asarray([cv_Lx[ipos],cv_Lx[ipos],cv_Lx[ipos],cv_Lx[ipos],cv_Lx[ipos]])
            px = xx+1
            py = yy+1 #yy+2
            fp = (py+1)*Nx+px
            cp = py*Nx+px
            lp = (py-1)*Nx+px
#            ffp = (py+2)*Nx+px
#            llp = (py-2)*Nx+px
            idy_row[counter:counter+3]= np.asarray([rpos,rpos,rpos])
            idy_col[counter:counter+3]= np.asarray([fp,cp,lp])
            idy_d1yA[counter:counter+3]= np.asarray([cv_Ly[ipos],cv_Ly[ipos],cv_Ly[ipos]])
#            idy_row[counter:counter+9] = np.asarray([rpos,rpos,rpos,rpos,rpos,rpos,rpos,rpos,rpos])   
#            idy_col[counter:counter+9] = np.asarray([fp-1,fp,fp+1,cp-1,cp,cp+1,lp-1,lp,lp+1])
#            idy_d1yA[counter:counter+9]= np.asarray([cv_Ly[ipos],cv_Ly[ipos],cv_Ly[ipos],cv_Ly[ipos],cv_Ly[ipos],cv_Ly[ipos],cv_Ly[ipos],cv_Ly[ipos],cv_Ly[ipos]])
#            idy_row[counter:counter+5] = np.asarray([rpos,rpos,rpos,rpos,rpos])   
#            idy_col[counter:counter+5] = np.asarray([ffp,fp,cp,lp,llp])
#            idy_d1yA[counter:counter+5]= np.asarray([cv_Ly[ipos],cv_Ly[ipos],cv_Ly[ipos],cv_Ly[ipos],cv_Ly[ipos]])
#            counter = counter+9
            counter = counter+3
#            counter = counter+5
            if ((rpos+1)%nx == 0):#if (rpos!= 0) and (rpos%nx == 0):
                cpos = cpos+2
    spmat_Lx = csc_matrix((idx_d1xA, (idx_row, idx_col)), shape=(nx*ny, (Nx)*(Ny))) 
    spmat_Ly = csc_matrix((idy_d1yA, (idy_row, idy_col)), shape=(nx*ny, (Nx)*(Ny)))
    mat_pux = (spmat_Lx.multiply(mat_d1xA))/(2*dx) 
    mat_puy = (spmat_Ly.multiply(mat_d1yA))/(2*dy)
    return mat_pux, mat_puy, spmat_Lx, spmat_Ly

def sparse_puxy_double(Lx, Ly, mat_d1xA, mat_d1yA, dx, dy, Nx, Ny):
    nx = Nx-2
    ny = Ny-2
    cv_Lx = np.reshape(Lx, (nx*ny,), order='F')
    cv_Ly = np.reshape(Ly, (nx*ny,), order='F')
    idx_row = np.zeros((3*nx*ny))
    idx_col = np.zeros((3*nx*ny))
    idx_d1xA = np.empty((3*nx*ny))#,dtype=complex)
    idy_row = np.zeros((3*nx*ny))
    idy_col = np.zeros((3*nx*ny))
    idy_d1yA = np.empty((3*nx*ny))#,dtype=complex)
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
            idx_d1xA[counter:counter+3]= np.asarray([cv_Lx[ipos],cv_Lx[ipos],cv_Lx[ipos]])
            px = xx+1
            py = yy+1 #yy+2
            fp = (py+1)*Nx+px
            cp = py*Nx+px
            lp = (py-1)*Nx+px
            idy_row[counter:counter+3]= np.asarray([rpos,rpos,rpos])
            idy_col[counter:counter+3]= np.asarray([fp,cp,lp])
            idy_d1yA[counter:counter+3]= np.asarray([cv_Ly[ipos],cv_Ly[ipos],cv_Ly[ipos]])
            counter = counter+3
            if ((rpos+1)%nx == 0):
                cpos = cpos+2
    spmat_Lx = csc_matrix((idx_d1xA, (idx_row, idx_col)), shape=(nx*ny, (Nx)*(Ny))) 
    spmat_Ly = csc_matrix((idy_d1yA, (idy_row, idy_col)), shape=(nx*ny, (Nx)*(Ny)))
    mat_pux = (spmat_Lx.multiply(mat_d1xA))/(dx**2) 
    mat_puy = (spmat_Ly.multiply(mat_d1yA))/(dy**2)
    return mat_pux, mat_puy, spmat_Lx, spmat_Ly

def sparse_puxy_5(mat_Lx, mat_Ly, mat_d1xA, mat_d1yA, dx, dy, Nx, Ny):
    nx = Nx-2
    ny = Ny-2
    cv_Lx = np.reshape(mat_Lx, (nx*ny), order='F')
    cv_Ly = np.reshape(mat_Ly, (nx*ny), order='F')
    idx_row = np.zeros((3*nx*ny))
    idx_col = np.zeros((3*nx*ny))
    idx_d1xA = np.empty((3*nx*ny))#,dtype=complex)
    idy_row = np.zeros((3*nx*ny))
    idy_col = np.zeros((3*nx*ny))
    idy_d1yA = np.empty((3*nx*ny))#,dtype=complex)
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
            idx_d1xA[counter:counter+3]= np.asarray([cv_Lx[ipos],cv_Lx[ipos],cv_Lx[ipos]])
            px = xx+1
            py = yy+1 #yy+2
            fp = (py+1)*Nx+px
            cp = py*Nx+px
            lp = (py-1)*Nx+px
#            ffp = (py+2)*Nx+px
#            llp = (py-2)*Nx+px
            idy_row[counter:counter+3]= np.asarray([rpos,rpos,rpos])
            idy_col[counter:counter+3]= np.asarray([fp,cp,lp])
            idy_d1yA[counter:counter+3]= np.asarray([cv_Ly[ipos],cv_Ly[ipos],cv_Ly[ipos]])
#            counter = counter+9
            counter = counter+3
#            counter = counter+5
            if ((rpos+1)%nx == 0):#if (rpos!= 0) and (rpos%nx == 0):
                cpos = cpos+2
    spmat_Lx = csc_matrix((idx_d1xA, (idx_row, idx_col)), shape=(nx*ny, Nx*Ny)) 
    spmat_Ly = csc_matrix((idy_d1yA, (idy_row, idy_col)), shape=(nx*ny, Nx*Ny))
    mat_pux = (spmat_Lx.multiply(mat_d1xA))/(2*dx) 
    mat_puy = (spmat_Ly.multiply(mat_d1yA))/(2*dy)
    return mat_pux, mat_puy, spmat_Lx, spmat_Ly
