## https://abagen.readthedocs.io/en/stable/user_guide/download.html#loading-the-ahba-data;
# pip install numpy==1.24.4 pandas==1.5.3 scipy nibabel
# pip install abagen==0.1.3
# pip install nilearn==0.11.1

import abagen
print(abagen.__file__)
# print("abagen版本：", abagen.__version__)  # 示例：0.1.3
from nilearn.datasets import fetch_atlas_destrieux_2009
import pandas as pd #pandas==1.5.3
import os
import nibabel as nib

files = abagen.fetch_microarray(donors='all', verbose=0)
print(files.keys())
print(sorted(files['9861']))

data = files['9861']
annotation = abagen.io.read_annotation(data['annotation'])
print(annotation)
probes = abagen.io.read_probes(data['probes'])
print(probes)

# get DK atlas
DK_atlas = abagen.fetch_desikan_killiany()
print(DK_atlas['image'])
print(DK_atlas['info'])

## ========== get destrieux atlas ==================
atlas_d = fetch_atlas_destrieux_2009(lateralized=True)  # lateralized=True 时左右半球分开标签
nifti_path = atlas_d.maps # ...\fengn\nilearn_data\destrieux_2009\destrieux2009_rois_lateralized.nii.gz
# lut = atlas_d.lut  # DataFrame，包含 index & regions's name
#
# # 写 CSV
# csv_path = "...\write_loading_back_25xin\enrichGene_writeBack\destrieux_lut.csv"
# lut.to_csv(csv_path, index=False)

## ============= 修复abagen================
# import pathlib
# path = pathlib.Path(r"...\write_loading_back_25xin\writeMgzBack\.venv\Lib\site-packages\abagen\probes_.py")
# text = path.read_text(encoding="utf-8")
# # 修复 inplace=False 错误
# fixed = text.replace("set_axis(symbols, axis=1, inplace=False)", "set_axis(symbols, axis=1)")
# path.write_text(fixed, encoding="utf-8")
# print("✅ 修复完成，请重新运行 get_expression_data()。")
#
# import abagen.probes_ as p
# print(p.collapse_probes.__code__.co_firstlineno)# 526

# ---------- 从 abagen 获取 Destrieux 表达（默认返回 ROI x genes） ----------
from abagen import get_expression_data

# 获取基因表达数据
expression = get_expression_data(
    atlas=nifti_path,
    lr_mirror='bidirectional',  
    return_donors=False
)
type(expression)
expression.shape # 148*15633， row: brain regions; column: gene expression;
# 脑区行对应destrieux_lut.csv
## under ...\write_loading_back_25xin\enrichGene_writeBack

# 保存 expression 为 CSV 文件
output_path = ".../write_loading_back_25xin/enrichGene_writeBack/Extracted_Expression.csv"
expression.to_csv(output_path, index=True)  # index=True 保留行名（脑区标签）

## get the reporting methods;
from abagen import reporting
generator = reporting.Report(atlas=nifti_path, lr_mirror='bidirectional', return_donors=False)
report = generator.gen_report()

# 保存为 txt 文件
output_path = ".../write_loading_back_25xin/enrichGene_writeBack/abagen_ROIgene_report.txt" 
with open(output_path, "w", encoding="utf-8") as f:
    f.write(report)

# pip install templateflow
from templateflow import api;
api.get('MNI152NLin2009cAsym')
api.get('OASIS30ANTs')

