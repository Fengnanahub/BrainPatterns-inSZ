# -*- coding: utf-8 -*-
"""
Created on Fri Dec 29 2023

@author: YOGA
"""

import numpy as np
from mvlearn.embed import CCA, MCCA, KMCCA
# from numpy.linalg import norm

from scipy.stats import pearsonr
import pandas as pd


def CalCorrelation(x, y, z):
    # threshold = 0.2
    correlation = []
    penalty = []
    for i in range(x.shape[1]):
       correlation12 = abs(pearsonr(x[:, i], y[:, i])[0])
       correlation13 = abs(pearsonr(x[:, i], z[:, i])[0])
       correlation23 = abs(pearsonr(y[:, i], z[:, i])[0])
       correlation.append(min(correlation12, correlation13, correlation23))
    return correlation


def CalPvalue(initial_correlation, permute_correlation):
    permute_correlation = np.array([[permute_correlation[i][j] for j in range(len(initial_correlation))] for i in
                                    range(len(permute_correlation))])
    initial_correlation = np.array([initial_correlation[i]for i in range(len(initial_correlation))])
    pvalue = (permute_correlation > initial_correlation).mean(axis=0)[0]
    return pvalue


def SplitHalf(x, y, z, permute=1000, train_ratio=0.5):
    # regularization value of .5 for each view
    mcca = MCCA(min(x.shape[1],y.shape[1], z.shape[1]), regs=0.5)
    train_index = np.random.choice(x.shape[0],int(train_ratio * x.shape[0]), replace = False)
    test_index = np.delete(np.arange(x.shape[0]),train_index)
    x_train = x[train_index,:]
    x_test = x[test_index,:]
    y_train = y[train_index,:]
    y_test = y[test_index,:]
    z_train = z[train_index,:]
    z_test = z[test_index,:]

    xyz_train = [x_train, y_train, z_train] # list
    mcca.fit(xyz_train)

    xyz_test = [x_test, y_test, z_test]
    # mcca_scores = mcca.transform(xyz_test)# 3*25*2
    # print(mcca.canon_corrs(mcca_scores))
    x_test_cca, y_test_cca, z_test_cca = mcca.transform(xyz_test) # each output: 25*2
    initial_correlation = CalCorrelation(x_test_cca, y_test_cca, z_test_cca)
    permute_correlation = []
    for j in list(range(permute)):
        y_test_cca_copy = y_test_cca.copy()
        np.random.shuffle(y_test_cca_copy)
        cur_correlation = CalCorrelation(x_test_cca, y_test_cca_copy, z_test_cca)
        permute_correlation.append(cur_correlation)
    pvalue = CalPvalue(initial_correlation, permute_correlation)
    return pvalue


def TrainTestCcaPvalue(x, y, z, permute=1000, nums=10, train_ratio=0.5):
    pvalues = []
    for i in range(nums):
        pvalue = SplitHalf(x, y, z, permute=permute, train_ratio=train_ratio)
        pvalues.append(pvalue)
    return pvalues


def sig_num_pvalue(pvalues):
    pvalues = np.array(pvalues)
    return (pvalues <= 0.05).sum(axis=0)


class BuildTree():
    def __init__(self, x, y, z, c, mtry=1 / 3, n_split=21, nodedepth=10, nodesize=80, permute=1000, nums=10,
                 train_ratio=0.5):
        """
        x:np.array,(n_samples,n_dimension1)
        y:np.array,(n_samples,n_dimension2)
        z:np.array,(n_samples,n_dimension3)
        x,y, z are 3 sets of variables used to find multivariate correlations
        c:np.array,(n_samples,n_dimension4),c is covariate for TGHCC division
        mtry:float,represents what proportion of covariates to choose from z to divide each time the parent node is divided
        n_split:int,Represents how many cutoffs are selected for each covariate selected for partitioning
        nodedepth:int,represents the maximum depth of the tree
        nodesize:int,Represents the maximum number of samples contained in each leaf node
        permute:int,the number of permutations included in the permutation test
        nums:int,Represents the number of split-half in the permutation test
        train_ratio:float((0,1)),The proportion of the training set when cca is fitted
        """
        self.x = x
        self.y = y
        self.z = z
        self.c = c
        self.mtry = mtry
        self.n_split = n_split
        self.nodedepth = nodedepth
        self.nodesize = nodesize
        self.permute = permute
        self.nums = nums
        self.train_ratio = train_ratio

    def DataSplit(self, train_x, train_y, train_z, train_c, index, split_point):
        left_index = np.where(train_c[:, index] < split_point)[0]
        right_index = np.where(train_c[:, index] >= split_point)[0]
        left = [train_x[left_index, :], train_y[left_index, :], train_z[left_index, :], train_c[left_index, :]]
        right = [train_x[right_index, :], train_y[right_index, :], train_z[right_index, :], train_c[right_index, :]]
        return left, right

    def SplitCriterionPermute(self, train_x, train_y, train_z):
        pvalues = TrainTestCcaPvalue(train_x, train_y, train_z, permute=self.permute, nums=self.nums,
                                     train_ratio=self.train_ratio)
        sig_p = sig_num_pvalue(pvalues)
        return sig_p

    def GetBestSplit(self, train_x, train_y, train_z, train_c):
        num_split_features = max(round(self.mtry * train_c.shape[1]), 1)
        select_features = np.random.choice(train_c.shape[1], num_split_features, replace=False)
        diff_cca = 0
        for index in select_features:
            # for each selected feature column， get the n_split feature cutoffs；
            all_splits = [np.percentile(train_c[:, index], k) for k in np.linspace(0, 100, self.n_split)]
            for split_point in all_splits:
                # for each cutoff;
                left, right = self.DataSplit(train_x, train_y, train_z, train_c, index, split_point)
                if left[0].shape[0] < self.nodesize or right[0].shape[0] < self.nodesize:
                    continue
                left_cca = self.SplitCriterionPermute(left[0], left[1], left[2]) # H-index
                right_cca = self.SplitCriterionPermute(right[0], right[1], right[2])
                cur_diff = max(np.abs(left_cca).max(), np.abs(right_cca).max()) * np.sqrt(
                    left[0].shape[0] * right[0].shape[0])
                if cur_diff > diff_cca:
                    b_index, b_value, diff_cca, b_left, b_right = index, split_point, cur_diff, left, right
        if diff_cca == 0:
            return {'index': None, 'split_point': None, 'left': [pd.DataFrame()], 'right': [pd.DataFrame()]}
        return {'index': b_index, 'split_point': b_value, 'left': b_left, 'right': b_right}

    def SubSplit(self, root, depth):
        left = root['left']
        right = root['right']
        del (root['left'])
        del (root['right'])
        if depth == self.nodedepth:
            root['left'] = left
            root['right'] = right
            return None
        root['left'] = self.GetBestSplit(left[0], left[1], left[2], left[3])
        root['right'] = self.GetBestSplit(right[0], right[1], right[2], right[3])
        if root['left']['left'][0].shape[0] == 0:  # or root['left']['right'][0].shape[0] < nodesize:
            root['left'] = left
        else:
            self.SubSplit(root['left'], depth + 1)
        if root['right']['left'][0].shape[0] == 0:  # or root['right']['right'][0].shape[0] < nodesize:
            root['right'] = right
        else:
            self.SubSplit(root['right'], depth + 1)

    def train(self):
        root = self.GetBestSplit(self.x, self.y, self.z, self.c)
        self.SubSplit(root, 1)
        return root