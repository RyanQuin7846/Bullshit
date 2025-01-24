# -*- coding: utf-8 -*-
"""
Created on Sat Oct 12 19:10:15 2024

@author: Server2
"""

import numpy as np
import math
from scipy.signal import savgol_filter


def complex_gradient(file):
    def filereader(file_name,z,y,x):
        # 用于存储分割后的数据
        data = []
        
        with open(file_name, 'r', encoding='utf-8') as file:
            # 跳过前20行
            for _ in range(20):
                next(file)
            
            # 从第21行开始读取并分割为9列
            for line in file:
                columns = line.strip().split()  # 使用空格分隔
                if len(columns) == 9:  # 确保每行有9列
                    try:
                        float_values = [float(value) for value in columns]
                        data.append(float_values)
                    except ValueError as e:
                        print(f"ValueError encountered: {e} for line: {line.strip()}")
                else:
                    print(f"Line doesn't have 9 columns: {line.strip()}")
        
            
        data_matrix = np.array(data)
        matrix_x = data_matrix[:,0].reshape(z,y,x)
        matrix_y = data_matrix[:,1].reshape(z,y,x)
        matrix_z = data_matrix[:,2].reshape(z,y,x)
        
        real_x = data_matrix[:,3].reshape(z,y,x)
        imag_x = data_matrix[:,4].reshape(z,y,x)
        
        real_y = data_matrix[:,5].reshape(z,y,x)
        imag_y = data_matrix[:,6].reshape(z,y,x)
        
        real_z = data_matrix[:,7].reshape(z,y,x)
        imag_z = data_matrix[:,8].reshape(z,y,x)
        
        return matrix_x, matrix_y, matrix_z, real_x, imag_x, real_y, imag_y, real_z, imag_z
    
    file_name = file
    matrix_x, matrix_y, matrix_z, h_real_x, h_imag_x, h_real_y, h_imag_y, h_real_z, h_imag_z = filereader(file_name,20,46,45)
    
    
    h_x = h_real_x + 1j*h_imag_x
    h_y = h_real_y + 1j*h_imag_y
    h_z = h_real_z + 1j*h_imag_z
    h = (h_x + 1j*h_y)/2
    
    slice = 0
    
    line_x = np.zeros(matrix_x.shape[2]-1)
    for i in range (0,matrix_x.shape[2]-1):
        line_x[i] = matrix_x[slice,0,i+1] - matrix_x[slice,0,i]
        step_x = np.mean(line_x)
        
    line_y = np.zeros(matrix_y.shape[1]-1)
    for i in range (0,matrix_y.shape[1]-1):
        line_y[i] = matrix_y[slice,i+1,0] - matrix_y[slice,i,0]
        step_y = np.mean(line_y)
        
    line_z = np.zeros(matrix_z.shape[0]-1)
    for i in range (0,matrix_y.shape[0]-1):
        line_z[i] = matrix_z[i+1,0,0] - matrix_z[i,0,0]
        step_z = np.mean(line_z)
    
    window_width = 3
    polynomial_order = 2
    
    h_real_gradient_x = savgol_filter(h.real, window_width, polynomial_order, deriv=1, delta=step_x, axis=-1, mode='wrap', cval=0.0)
    h_imag_gradient_x = savgol_filter(h.imag, window_width, polynomial_order, deriv=1, delta=step_x, axis=-1, mode='wrap', cval=0.0)
    h_gradient_x = h_real_gradient_x + 1j*h_imag_gradient_x
    
    h_real_gradient_y = savgol_filter(h.real, window_width, polynomial_order, deriv=1, delta=step_y, axis=1, mode='wrap', cval=0.0)
    h_imag_gradient_y = savgol_filter(h.imag, window_width, polynomial_order, deriv=1, delta=step_y, axis=1, mode='wrap', cval=0.0)
    h_gradient_y = h_real_gradient_y + 1j*h_imag_gradient_y
    
    h_real_gradient_z = savgol_filter(h.real, window_width, polynomial_order, deriv=1, delta=step_z, axis=0, mode='wrap', cval=0.0)
    h_imag_gradient_z = savgol_filter(h.imag, window_width, polynomial_order, deriv=1, delta=step_z, axis=0, mode='wrap', cval=0.0)
    h_gradient_z = h_real_gradient_z + 1j*h_imag_gradient_z
    
    h_real_laplacian_x = savgol_filter(h.real, window_width, polynomial_order, deriv=2, delta=step_x, axis=-1, mode='wrap', cval=0.0)
    h_imag_laplacian_x = savgol_filter(h.imag, window_width, polynomial_order, deriv=2, delta=step_x, axis=-1, mode='wrap', cval=0.0)
    h_laplacian_x = h_real_laplacian_x + 1j*h_imag_laplacian_x
    
    h_real_laplacian_y = savgol_filter(h.real, window_width, polynomial_order, deriv=2, delta=step_y, axis=1, mode='wrap', cval=0.0)
    h_imag_laplacian_y = savgol_filter(h.imag, window_width, polynomial_order, deriv=2, delta=step_y, axis=1, mode='wrap', cval=0.0)
    h_laplacian_y = h_real_laplacian_y + 1j*h_imag_laplacian_y
    
    h_real_laplacian_z = savgol_filter(h.real, window_width, polynomial_order, deriv=2, delta=step_z, axis=0, mode='wrap', cval=0.0)
    h_imag_laplacian_z = savgol_filter(h.imag, window_width, polynomial_order, deriv=2, delta=step_z, axis=0, mode='wrap', cval=0.0)
    h_laplacian_z = h_real_laplacian_z + 1j*h_imag_laplacian_z
    
    h_z_real_gradient_x = savgol_filter(h_z.real, window_width, polynomial_order, deriv=1, delta=step_x, axis=-1, mode='wrap', cval=0.0)
    h_z_imag_gradient_x = savgol_filter(h_z.imag, window_width, polynomial_order, deriv=1, delta=step_x, axis=-1, mode='wrap', cval=0.0)
    h_z_gradient_x = h_z_real_gradient_x + 1j*h_z_imag_gradient_x
    
    h_z_real_gradient_y = savgol_filter(h_z.real, window_width, polynomial_order, deriv=1, delta=step_y, axis=1, mode='wrap', cval=0.0)
    h_z_imag_gradient_y = savgol_filter(h_z.imag, window_width, polynomial_order, deriv=1, delta=step_y, axis=1, mode='wrap', cval=0.0)
    h_z_gradient_y = h_z_real_gradient_y + 1j*h_z_imag_gradient_y
    
    h_z_real_gradient_z = savgol_filter(h_z.real, window_width, polynomial_order, deriv=1, delta=step_z, axis=0, mode='wrap', cval=0.0)
    h_z_imag_gradient_z = savgol_filter(h_z.imag, window_width, polynomial_order, deriv=1, delta=step_z, axis=0, mode='wrap', cval=0.0)
    h_z_gradient_z = h_z_real_gradient_z + 1j*h_z_imag_gradient_z
    
    beta_x = 2*(1/2)*(h_gradient_x-1j*h_gradient_y) + (1/2)*h_z_gradient_z
    beta_y = 1j*(2*(1/2)*(h_gradient_x-1j*h_gradient_y) + (1/2)*h_z_gradient_z)
    beta_z = h_gradient_z - (1/2)*(h_z_gradient_x+1j*h_z_gradient_y)
    
    h_laplacian = h_laplacian_x + h_laplacian_y + h_laplacian_z
    
    return beta_x, beta_y, beta_z, h_laplacian