# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 20:36:48 2024

@author: Server2
"""

import numpy as np

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

file_name = 'data/raw data/4mm tumor pos1/E.txt'
matrix_x, matrix_y, matrix_z, e_real_x, e_imag_x, e_real_y, e_imag_y, e_real_z, e_imag_z = filereader(file_name,20,46,45)

file_name = 'data/raw data/4mm tumor pos1/J.txt'
_, _, _, j_real_x, j_imag_x, j_real_y, j_imag_y, j_real_z, j_imag_z = filereader(file_name,20,46,45)

file_name = 'data/raw data/4mm tumor pos1/D.txt'
_, _, _, d_real_x, d_imag_x, d_real_y, d_imag_y, d_real_z, d_imag_z = filereader(file_name,20,46,45)

file_name = 'data/raw data/4mm tumor pos1/H.txt'
_, _, _, h_real_x, h_imag_x, h_real_y, h_imag_y, h_real_z, h_imag_z = filereader(file_name,20,46,45)

e_x = e_real_x + 1j*e_imag_x
e_y = e_real_y + 1j*e_imag_y
e_z = e_real_z + 1j*e_imag_z
e = e_x + 1j*e_y

j_x = j_real_x + 1j*j_imag_x
j_y = j_real_y + 1j*j_imag_y
j_z = j_real_z + 1j*j_imag_z
j = j_x + 1j*j_y

d_x = d_real_x + 1j*d_imag_x
d_y = d_real_y + 1j*d_imag_y
d_z = d_real_z + 1j*d_imag_z
d = d_x + 1j*d_y

h_x = h_real_x + 1j*h_imag_x
h_y = h_real_y + 1j*h_imag_y
h_z = h_real_z + 1j*h_imag_z
h = (h_x + 1j*h_y)/2

e0 = 8.8541878176e-12

condy = np.real(j/e)    
permy = np.real((d/e)/e0)

np.save("data/raw data/4mm tumor pos1 v2/X.npy", matrix_x)
np.save("data/raw data/4mm tumor pos1 v2/Y.npy", matrix_y)
np.save("data/raw data/4mm tumor pos1 v2/Z.npy", matrix_z)
np.save("data/raw data/4mm tumor pos1 v2/H.npy",h)
np.save("data/raw data/4mm tumor pos1 v2/condy.npy", condy)
np.save("data/raw data/4mm tumor pos1 v2/permy.npy", permy)