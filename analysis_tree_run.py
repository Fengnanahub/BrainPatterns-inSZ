# -*- coding: utf-8 -*-
"""
Created on Thu Jun 30 09:44:01 2022

@author: lenovo
"""
## here saved the main code for multi-times tree analysis;

import numpy as np
import pandas as pd
from RFHmCCA import BuildTree
import pickle
from RFHmCCA import TrainTestCcaPvalue, sig_num_pvalue

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
    return data_tree

def GetIndex(element, train_x, train_y, train_z):
    merged_data = pd.concat([train_x, train_y, train_z], axis=1)
    merged_dataVar = merged_data.values
    row_index = np.where(np.all(merged_dataVar == element, axis=1))[0]
    return row_index
    # data_list = [list(data_list.values[i,:]) for i in range(data_list.shape[0])]
    # element = list(element)
    # return train_x.index[data_list.index(element)]

def GetMembership(data_tree):
    """
    data_tree:list,result from analysis_tree
    membership:dataframe,each subject in train_x, train_y, train_z belong to which leaf node
    """
    membership = pd.DataFrame([0] * train_x.shape[0],index = train_x.index,columns = ["node"])
    for i in range(len(data_tree)):
        for j in range(data_tree[i][0].shape[0]):
            yj = data_tree[i][1][j,:]
            y_transpose = np.transpose(yj.reshape(len(yj), 1))
            xj = data_tree[i][0][j,:]
            xj_org = xj.reshape(len(xj), 1)
            zj = data_tree[i][2][j,:]
            zj_org = zj.reshape(len(zj), 1)
            elementj = np.concatenate((xj_org, y_transpose, zj_org), axis=1)

            index = GetIndex(elementj, train_x, train_y, train_z)
            membership.loc[index] = (i + 1)
    return membership

## after RFHmCCA tree train and membership
# find Homogeneous leaf node as subgroup candidate (for many trees)
def compute_mcca(x_subset, y_subset, z_subset):
    mcca_pvalue = TrainTestCcaPvalue(x_subset, y_subset, z_subset, permute=1000, nums=10, train_ratio=0.5)
    mcca_sig_p = sig_num_pvalue(mcca_pvalue)
    return mcca_sig_p

def HomoSubgroupDetect(x, y, z, membership):
    unique_values = membership['node'].unique()
    # loop every leaf node
    results_leafHomo = []
    for leafi in range(len(unique_values)):
        # 根据 member 的值选择相应的行
        subset_indices = np.where(membership == unique_values[leafi])[0]
        x_subset = x[subset_indices, :]
        y_subset = y[subset_indices, :]
        z_subset = z[subset_indices, :]

        HomeIndex = compute_mcca(x_subset, y_subset, z_subset)

        print(f"mCCA result for subgroup {unique_values[leafi]}: {HomeIndex}")

        result_entry = {
            'leafIndex': unique_values[leafi],
            'SubsetIndices': subset_indices.tolist(),
            'HomeIndex': HomeIndex.tolist()
        }
        results_leafHomo.append(result_entry) # list,length = num of leaf node;
    return results_leafHomo

## find whether there is leaf node with H =10;
def HomoSubgroupGet(tree, membership):
    # get variable saved in tree
    x= tree.x
    y= tree.y
    z= tree.z

    leafHomo = HomoSubgroupDetect(x, y, z, membership)
    leafIndex = []
    subsetIndices = []
    for entry in leafHomo:
        if entry['HomeIndex'] == 10:
            leafIndex.append(entry['leafIndex']) # which leaf node in given tree;
            subsetIndices.append(entry['SubsetIndices']) # subjects ID index in this Homo =10 leaf node;
            print(f"leafIndex: {leafIndex}")
            print(f"Subset Indices: {subsetIndices}")
    return leafIndex, subsetIndices


num_runs = 1000
for run_number in range(1, num_runs + 1):

    path = r".../pythonProject_mCCA"
    # path = os.path.abspath(os.curdir)
    #  None colume in .csv is the sub index; DataFrame
    train_x = pd.read_csv(path + "/Train_x.csv", index_col=None) # 975*1
    train_y = pd.read_csv(path + "/Train_y.csv", index_col=None) # 975*166
    train_z = pd.read_csv(path + "/Train_z.csv", index_col=None) # 975*1
    train_c = pd.read_csv(path + "/Train_c.csv", index_col=None) # 975*7

    # delete the NA row in x, y, z, c
    allCom_col = np.hstack((train_x, train_y, train_z, train_c))  # colume combine
    nasumcol = np.isnan(allCom_col).sum(axis=1) # sum NA by row
    xyzc_use = allCom_col[nasumcol<1] # select noNA row,975, array
    subN = xyzc_use.shape[0]

    usex_values = xyzc_use[:, :train_x.shape[1]]
    usey_values = xyzc_use[:, train_x.shape[1]:train_x.shape[1]+train_y.shape[1]]
    usez_values = xyzc_use[:, train_x.shape[1]+train_y.shape[1]:train_x.shape[1]+train_y.shape[1]+train_z.shape[1]]
    usec_values = xyzc_use[:, xyzc_use.shape[1]-train_c.shape[1]: xyzc_use.shape[1]]

    tree = BuildTree(usex_values, usey_values, usez_values, usec_values)
    tree_root = tree.train()
    data_tree = AnalysisTree(tree_root) # list N (saved x,y,z,c in each leaf node);
    membership = GetMembership(data_tree) # 975*1, index of leaf node for each sub;

    unique_values = membership['node'].unique()
    print(unique_values)

    print(f'Finish Tree trained for run {run_number}')

    Homoresults = HomoSubgroupGet(tree, membership) # return 2 items;

    filtered_results = []
    # find subset_indices whether null (exist Homo leaf node);
    if Homoresults[0]:
        Leaf_Index = Homoresults[0]
        Subset_indices = Homoresults[1]
        filtered_results.append({'LeafIndex': Leaf_Index, 'SubsetIndices': Subset_indices})

        # save tree results while there is at least 1 leaf node has H = 10;
        filename = f'.../pythonProject_mCCA/store_Imagen_tree{run_number}.pckl'
        f = open(filename, 'wb')
        pickle.dump((tree, tree_root, data_tree, membership), f)
        f.close()
        print(f'Saved Tree for run {run_number} to {filename}')

        # save the Homo leaf index and subset index of tree{run_number};
        filename1 = f'.../pythonProject_mCCA/store_HomeRes_tree{run_number}.pckl'
        f = open(filename1, 'wb')
        pickle.dump(filtered_results, f)
        f.close()
        print(f'Saved Home node for tree{run_number} to {filename1}')

# filename = f'store1_IMAGEN_Traintree.pckl'
# with open(filename, 'rb') as f:
#     tree, tree_root, data_tree, membership = pickle.load(f)

