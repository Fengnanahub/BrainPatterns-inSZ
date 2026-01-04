# -*- coding: utf-8 -*-
"""
Created on Thu Jun 30 09:18:17 2022

@author: lenovo
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
import pickle

def get_params(x, y):
    x = sm.add_constant(x)
    md = sm.OLS(y, x)
    mdf = md.fit()
    return mdf.params
# print(mdf.summary())

path = r".../.csv"
# 977*269
brainarea_data_clean = pd.read_csv(path, index_col=0)
site_name = brainarea_data_clean.loc[:, "site"].value_counts().index
site_dict = dict(zip(list(site_name), np.arange(len(site_name))))
# site [2, 5, 7, 8, 1, 6, 4, 3], changed to [0, 1, 2, 3, 4, 5, 6, 7];
brainarea_data_clean.loc[:, "site"] = brainarea_data_clean.loc[:, "site"].map(site_dict)

colindex1 = brainarea_data_clean.columns.get_loc("Left_Lateral_Ventricle") # 47
colindex2 = brainarea_data_clean.columns.get_loc("EstimatedTotalIntraCranialVol") # 66
colindex3 = brainarea_data_clean.columns.get_loc("Left_G_S_frontomargin") # 76
colindex4 = brainarea_data_clean.columns.get_loc("Right_S_temporal_transverse") # 223, col in python , 0, 1, 2,...268;

# brain area volume， mm3 --> ml. 1137*67
brainarea_data_clean.iloc[:, 47:67] /= 1000
brainarea_data_clean.iloc[:, 76:224] /= 1000
print(list(brainarea_data_clean))
# save to csv ( site to dict, volume mm3 to ml)
brainarea_data_clean.to_csv('D:/DATA_Fudan/IMAGEN/IMAGEN_FS_FU2ana2403/LEQ_Brain_sym_JJsubgroup/dataComFU2_etc_szPRS977_269.csv', index=False)

# select the image data
cols_to_select = [1, 2, 4] + list(range(47, 67)) + list(range(76, 224))
TCdata = brainarea_data_clean.iloc[:, cols_to_select].copy() # 977*171

# Adding a new column 'age2' which is the square of 'age'
TCdata.loc[:, "age2"] = TCdata.loc[:, "age"] ** 2 # 977*172

TCdata = TCdata.loc[TCdata.loc[:, "Gender_Male"].notna(), :] # 977
TCdata = TCdata.loc[TCdata.loc[:, "age"].notna(), :] # 977
TCdata = TCdata.loc[TCdata.loc[:, "EstimatedTotalIntraCranialVol"].notna(), :] # 977
print(list(TCdata))

## ########## 166 brain region vol --->has na in column of Left_S_interm_prim_Jensen;
Imagecol = TCdata.iloc[:,3:171] # 977*168
has_nan = np.isnan(Imagecol).any()
# 查找具有 NaN 值的列的索引
cols_with_nan = Imagecol.columns[has_nan].tolist() #['Left_S_interm_prim_Jensen']
has_nan_row = Imagecol[cols_with_nan].isnull().any(axis=1)
Imagecol_NA = Imagecol[has_nan_row] # 2*168
# 使用布尔索引找出不含有 NaN 值的行
no_nan_row = ~Imagecol[cols_with_nan].isnull().any(axis=1)
Imagecol_clear = Imagecol[no_nan_row] # 975*168

TCdata_clean = TCdata[no_nan_row] # 975*172
brainarea_data_clean1 = brainarea_data_clean[no_nan_row] # 975*269
brainarea_data_clean1.to_csv('.../.csv', index=False)


# volumn ~ f(Age + Gender_Male + site +Age2+ TIV+1)
# regress age, sex, site, TIV from brain volume
TCdata = TCdata_clean.copy() # 975*172
colindex1 = TCdata.columns.get_loc("Left_Lateral_Ventricle") # 3
colindex2 = TCdata.columns.get_loc("Right_Accumbens_area") # 20
colindex3 = TCdata.columns.get_loc("Left_G_S_frontomargin") # 23
colindex4 = TCdata.columns.get_loc("Right_S_temporal_transverse") #170

fit_params = {}
print(list(TCdata.columns[3:21]) + list(TCdata.columns[23:171]))
for i in (TCdata.columns[3:21].union(TCdata.columns[23:171])):
    fit_params[i] = get_params(TCdata.loc[:,["age",'Gender_Male','site','age2', 'EstimatedTotalIntraCranialVol']],TCdata.loc[:,str(i)])

# apply--> residual vol for sub
diffdata = TCdata.copy()
for i in (diffdata.columns[3: 21].union(TCdata.columns[23:171])):
    diffdata.loc[:,i] -= fit_params[i].loc["const"] + np.dot(diffdata.loc[:,["age",'Gender_Male','site','age2', 'EstimatedTotalIntraCranialVol']].values,fit_params[i].iloc[1:].values)
cols_brain =list(range(3, 21)) + list(range(23, 171))
diffdata = diffdata.iloc[:, cols_brain].copy() # 975*166 brain region;
diffdata.to_csv('.../.csv', index=False)

# save regression model for each brain area(dict)
f = open('store_fitparams975_vol1.pckl', 'wb')
pickle.dump(fit_params, f)
f.close()

########## adjust 1. NLE; 2.PSY;
path1 = r"D:/DATA_Fudan/IMAGEN/IMAGEN_FS_FU2ana2403/LEQ_Brain_sym_JJsubgroup/dataComFU2_etc_szPRS_s975_269.csv"
brainarea_data_clean = pd.read_csv(path1, index_col=None) # 975*269

brainarea_data_clean.loc[:, "age2"] = brainarea_data_clean.loc[:, "age"] ** 2 # 975*270

colindex = brainarea_data_clean.columns.get_loc("fam_acc_dist_everFreq_sum") # 225
colindexx = brainarea_data_clean.columns.get_loc("CAPE_Fr_sum") # 268

fit_params1 = {}
print(brainarea_data_clean.columns[225])
for i in (brainarea_data_clean.columns[225:226].union(brainarea_data_clean.columns[268:269])):
    fit_params1[i] = get_params(brainarea_data_clean.loc[:,["age",'Gender_Male','site','age2', 'EstimatedTotalIntraCranialVol']],brainarea_data_clean.loc[:,str(i)])

# apply--> residual vol for sub
diffdataXZ = brainarea_data_clean.copy()
for i in (diffdataXZ.columns[225:226].union(diffdataXZ.columns[268:269])):
    diffdataXZ.loc[:,i] -= fit_params1[i].loc["const"] + np.dot(diffdataXZ.loc[:,["age",'Gender_Male','site','age2', 'EstimatedTotalIntraCranialVol']].values,fit_params1[i].iloc[1:].values)

diffdataX = diffdataXZ.iloc[:, 225].copy() # 166 brain region;
diffdataX.to_csv('.../.csv', index=False)

diffdataZ = diffdataXZ.iloc[:, 268].copy() # 166 brain region;
diffdataZ.to_csv('.../.csv', index=False)

# save regression model for each brain area(dict)
f = open('store_fitparams975_NLE_PSY.pckl', 'wb')
pickle.dump(fit_params1, f)
f.close()


def determine_num_components(s, desired_energy_percentage=0.80):
    total_energy = np.sum(s ** 2)
    cumulative_energy = np.cumsum(s ** 2) / total_energy
    num_components = np.argmax(cumulative_energy >= desired_energy_percentage) + 1
    return num_components

######## organize COV;----------- 1. diff 166 svd;
path2 = r".../.csv"
# 975*166
diffdata = pd.read_csv(path2, index_col=None)
print(list(diffdata))

imageRes = diffdata.copy() # 975*166
u,s,v = np.linalg.svd(imageRes)  ## 977 subjects, have NA in vol， SVD did not converge；here 975，No NA in vol;
# data: 975*166, u: 975*975,s, 166*1, v:166*166
num_components = determine_num_components(s, desired_energy_percentage=0.80) # 29
train_y_svd = np.dot(u[:,:29], np.diag(s[:29])) # dim reduced to 29 dims, 975*29

f = open('svdALL_res_s975vol166.pckl', 'wb')
pickle.dump((u, s, v), f)
f.close()

# save train_y SVD.
train_y_svd = pd.DataFrame(train_y_svd)
train_y_svd.to_csv('..../.csv', index=False)


######## organize COV; ---------- 2. SZprs svd;
path3 = r".../.csv"
# 975*269
dataTot = pd.read_csv(path3, index_col=None)
print(list(dataTot))

PRS_column = ['SZprs0p001_Zscore','SZprs0p01_Zscore','SZprs0p05_Zscore', 'SZprs0p1_Zscore', 'SZprs0p2_Zscore', 'SZprs0p3_Zscore', 'SZprs0p4_Zscore', 'SZprs0p5_Zscore']
szPRS = dataTot.loc[:, PRS_column] # 975*8
u,s,v = np.linalg.svd(szPRS)
# data: 975*6, u: 975*975,s, 6*1, v:6*6
num_components = determine_num_components(s, desired_energy_percentage=0.80) # 1
train_y_PRS_svd = np.dot(u[:,:1], np.diag(s[:1])) # dim reduced to 2 dims, 975*2

# save train_y SVD.
train_y_PRS_svd = pd.DataFrame(train_y_PRS_svd)
train_y_PRS_svd.to_csv('.../.csv', index=False)

f = open('svdALL_res_s975_szPRS.pckl', 'wb')
pickle.dump((u, s, v), f)
f.close()

######## organize COV; -------- 3. diff166 svd + SZprs svd--> train y;
path3 = r".../.csv"
# 975*269
diff166_svd = pd.read_csv(path3, index_col=None) # 975*29

path4 = r".../.csv"
# 975*2
PRS_svd = pd.read_csv(path4, index_col=None) # 975*1

train_y_comSvd = np.hstack((diff166_svd, PRS_svd)) # 975*30, C

column_names = ['diff166svd' + str(i+1) for i in range(29)] + ['szPRSsvd1']
train_y_comSvd = pd.DataFrame(train_y_comSvd, columns=column_names)
train_y_comSvd.to_csv(.../.csv', index=False)
