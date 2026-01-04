# -*- coding: utf-8 -*-
"""
Created on Mon Jan  9 23:15:44 2023

@author: lenovo
"""

from RFHmCCA import BuildTree
import numpy as np
import pickle

def AnalysisTree(tree):
    """
    tree:dict,result from TGHCC
    data_tree:list,store each leaf node from left to right in a tree
    """
    data_tree = []
    def GetTree(tree):
        if type(tree["left"]) == list:
            data_tree.append(tree["left"])
        else:
            GetTree(tree["left"])
        if type(tree["right"]) == list:
            data_tree.append(tree["right"])
        else:
            GetTree(tree["right"])
    GetTree(tree)
    return data_tree # list

num_simulated = 100
n_sample = 500
Npara = 0.01 # X,Z para for noise item; 0：noNoise； 0.01；
N1para = Npara/99 # Y para for noise item of each dim;
ans_left = []
for i in range(num_simulated):
    # X1, n_sample*2; Y1, n_sample*2; Z1, n_sample*2; second column = 1- first column;
    # X2, n_sample*2; Y2, n_sample*2; Z2, n_sample*2; multi variable normal distribution.
a
    correlation_12 = 0.9  # 
    correlation_13 = 0.95  # 
    correlation_23 = 0.85  # 
  
    cov_matrix = np.array([[1, correlation_12, correlation_13],
                           [correlation_12, 1, correlation_23],
                           [correlation_13, correlation_23, 1]])
   
    mean = [0, 0, 0]  # 设定均值为00.9
    Upara = np.random.multivariate_normal(mean, cov_matrix, size=n_sample) # u1, u2, u3;latent variable;

    # [A, C, B1,B2,...B99]
    Beta = np.linspace(0, 1, 102)
    A = Beta[1]
    C = Beta[2]

    cov_matrix_noise = np.array([[1, 0, 0],
                           [0, 1, 0],
                           [0, 0, 1]])

    mean_noise = [0, 0, 0]  # 设定均值为0
    Noise = np.random.multivariate_normal(mean_noise, cov_matrix_noise, size=n_sample) # noise1, noise2, noise3;

    X1 = np.zeros((500, 2))
    X1[:, 0] = np.random.normal(0, 1, 500)# X0
    X1[:, 1] = Upara[:, 0] - A*X1[:, 0]+ Npara*Noise[:, 0]# X1 = U1-A*X0+N*noise1;

    Y1 = np.zeros((500, 100))
    Y1[:, 0] = np.random.normal(0, 1, 500)
    # 计算y1到y99, i= 0-98;
    for i in range(99):
        Y1[:, i+1] = Upara[:, 1] - Beta[i+3] * Y1[:, 0]+ N1para*Noise[:, 1]

    Z1 = np.zeros((500, 2))
    Z1[:, 0] = np.random.normal(0, 1, 500) # Z0
    Z1[:, 1] = Upara[:, 2] - C*Z1[:, 0]+ Npara*Noise[:, 2] # Z1 = U3-C*Z0+N*noise3;

    X2 = np.absolute(np.random.multivariate_normal((0, 0), [[1, 0], [0, 1]], size=(n_sample,)))
    # Y2 = np.absolute(np.random.multivariate_normal((0, 0), [[1, 0], [0, 1]], size=(n_sample,)))
    Y2 = np.random.normal(0, 1, (n_sample, 100))
    Z2 = np.absolute(np.random.multivariate_normal((0, 0), [[1, 0], [0, 1]], size=(n_sample,)))

    x_all = np.vstack((X1, X2))
    y_all = np.vstack((Y1, Y2))
    z_all = np.vstack((Z1, Z2))

    # z = np.array(list(np.random.uniform(0,0.5,n_sample)) + list(np.random.uniform(0.5,1,n_sample)))
    # z_all = z.reshape(-1,1)
    c_all = np.absolute(np.random.normal(0, 1, (n_sample*2, 1)))
    
    sim_tree = BuildTree(x_all, y_all, z_all, c_all)
    sim_root = sim_tree.train()
    # ans.append(sim_root["left"][0].sum(axis = 1).sum())

    data_treeLeft = AnalysisTree(sim_root["left"])  # list
    nodeSubLeft = 0
    for j in range(len(data_treeLeft)):
        # fnn, data_treeLeft[j][0]: x data in leaf node j
        nodeSubLeft += data_treeLeft[j][0].shape[0]
    ans_left.append(nodeSubLeft)

f = open('store_simu_ansLeft_Noise.pckl', 'wb')
pickle.dump(ans_left, f)
f.close()
print(f"simu_ansLeft: {ans_left}")

