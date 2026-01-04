
################ write back directly ##############

import nibabel as nib
import pandas as pd
import numpy as np

# ==== 输入部分 ====
#aparc+aseg 模板 (subject/mri)
aparc_aseg_file = f".../write_loading_back_25xin/aparc.a2009s+aseg.mgz"

# CSV, ROI name+value
roi_csv = f".../allSubgroup4_y166roi_loadingORG.csv"

# output
output_file = f".../write_loading_back_25xin/subgroup4_back_Loading_all.mgz"

# ==== 读取数据 ====
img = nib.load(aparc_aseg_file)
data = img.get_fdata().astype(np.float32) #get template data

roi_df = pd.read_csv(roi_csv) # get loading文件
# roi_df.iloc[:, 2:6] = roi_df.iloc[:, 2:6].astype(np.float32) # 第3-6列的loading 数据格式为float 32; 防止freesurfer可视化时候报错；

# CSV应至少有两列: "ROI" (名称, 如 "Left-Hippocampus") 和 "Value" (统计值)
# ROI 名称要和 FreeSurferColorLUT 中的 label 名称一致
roi_dict = dict(zip(roi_df["ROIregion166_matchMgz"], roi_df["subgroup4"])) #e.g.  back subgroup1 -------------- need revise------------

# ==== 读取 LUT (用 FreeSurfer 的) ====
lut_file = f".../write_loading_back/FreeSurferColorLUT.txt"  
lut_dict = {}
with open(lut_file, "r") as f:
    for line in f:
        if line.startswith("#") or line.strip() == "":
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        idx, name = int(parts[0]), parts[1]
        lut_dict[name] = idx

# ==== create result====
result_data = np.zeros_like(data)

for roi_name, value in roi_dict.items():
    if roi_name not in lut_dict:
        print(f"[Warning] ROI {roi_name} not found in LUT, skipped")
        continue
    label_id = lut_dict[roi_name]
    mask = data == label_id
    result_data[mask] = value

# ==== save ====
result_img = nib.Nifti1Image(result_data, affine=img.affine, header=img.header)
nib.save(result_img, output_file) # write back

print(f"✅ Done! Result saved to {output_file}")

############## visualize 
from nilearn import plotting

img = nib.load(output_file)

# 转换数据格式folat 64 为float 32;
data = img.get_fdata()
data = data.astype('float32')
new_img = nib.Nifti1Image(data, img.affine)

## ----------------------------------------- need revise-----------------------------
saveHTML_path = r".../write_loading_back_25xin/subgroup4_back_Loading_all.html"

view = plotting.view_img(new_img, threshold=None)
view.save_as_html(saveHTML_path)

