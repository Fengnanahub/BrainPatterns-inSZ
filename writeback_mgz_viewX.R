##  mcca project resulted loading or contribution (based on a2009s+aseg),view;

# =====================loading view ===========================
rm(list = ls())
R.version.string # 查看R版本
# R4.5.1, Tools → Global Options → General设置；切换R version，all packages need to be installed again.

# install.packages("ggplot2", type = "binary")
library(ggplot2)
library(tidyverse)
#install.packages("ggseg")
library(ggseg)
library(ggseg3d)
library(sf)
library(ggrepel)
library(dplyr)
library(float)

# install.packages(c("sf", "sp", "ggseg3d", "rgl", "neurobase", "jsonify"), type = "binary")
library(sp)
library(rgl)
library(neurobase)
library(jsonify)

library(remotes)
library(ggsegDKT)

# =========== install aseg package
# # Enable this universe
# options(repos = c(
#   ggseg = 'https://ggseg.r-universe.dev',
#   CRAN = 'https://cloud.r-project.org'))
# 
# # Install some packages
# install.packages('ggsegDefaultExtra')
library(ggsegDefaultExtra)

# =========== install Desterieux a2009s package
# Enable this universe
# options(repos = c(
#   ggseg = 'https://ggseg.r-universe.dev',
#   CRAN = 'https://cloud.r-project.org'))
# 
# # Install some packages
# install.packages('ggsegDesterieux')
library(ggsegDesterieux)
ggsegDesterieux::desterieux
ggsegDesterieux::desterieux_3d
# ---------------------- 依赖安装完成


# ======================
## ---------- 加载数据：
# ======================
region_loading_all <- read.table('.../write_loading_back_25xin/allSubgroup4_y166roi_loadingORG.csv',
                                 head = TRUE, sep=",") # 191*6
colnames(region_loading_all)[2] <- "label" # 与aparc a2009s模板一致；

# 去掉前缀 "ctx_"
region_loading_all$label[44:nrow(region_loading_all)] <- sub("^ctx_", "", region_loading_all$label[44:nrow(region_loading_all)])

datause <- region_loading_all[, c(2,3)] # 191*2, select subgroup x-----------
colnames(datause)[2] <- "value"

# datause_filtered <- datause %>%
#   filter(!is.na(value) & abs(value) >= 0.02) # G1, 34*2; G2, 51*2;G3，42；G4：43；
datause_filtered <- datause 

# ============ organize data
data_cortical <- datause_filtered %>%
  filter(label %in% desterieux$data$label) # G1, 29;G2, 46;g3,41;G4, 41;

data_subcortical <- datause_filtered %>%
  filter(label %in% hcpa_3d$ggseg_3d[[1]]$label)# G1, 5; G2, 5;g3,1;G4, 2;

ls("package:ggsegDesterieux")
# [1] "desterieux"    "desterieux_3d"

ls("package:ggsegDefaultExtra")
# [1] "dkextra" "hcpa_3d"

desterieux$data # csv 皮质数据相对于desterieux， 皮质label名字，多ctx_前缀(因此提前去掉)；
desterieux_3d$ggseg_3d # csv 皮质数据相对于desterieux_3d， 皮质label名字，多ctx_前缀(因此提前去掉)；
hcpa_3d$ggseg_3d #  hcpa_3d对应aseg亚皮质，csv 亚皮质数据相对于aseg, label，一致；

#============  2d皮质
# 参考 https://blog.csdn.net/lazysnake666/article/details/125967171；
ats_info=desterieux
someData = tibble(
  label = data_cortical$label,
  p = data_cortical$value)
someData %>% 
  brain_join(ats_info) %>% 
  reposition_brain(hemi ~ side) %>% 
  ggplot() + 
  geom_sf(aes(fill = p), colour = "black", alpha = 0.6) +  # 脑区分界黑色 # alpha 越小，颜色越淡。
  scale_fill_viridis_c(option = "plasma", # 颜色，turbo, cividis，"viridis", "magma", "inferno", "plasma"
                       na.value = "white",# NA 显示白色
                       direction = 1) +  
  theme_void() # 去掉背景
# 


# 画 3D 脑图（亚皮质）=== at least two rows data;========
library(RColorBrewer)
pal <- colorRampPalette(rev(brewer.pal(11, "PuOr")))(100) # "RdBu"
p<- ggseg3d(
  .data = data_subcortical,
  atlas = hcpa_3d,
  label = "label",       # 告诉 ggseg3d 用 label 匹配
  colour = "value",      # 上色列,"value";
  hemisphere = "subcort",   # 关键：只绘制 subcortical
  palette = pal, # 否则定义为=pal;
  na.colour = "lightgrey",   # 没值的区域显示白色
  na.alpha = 0.3,           # 更透明
  show.legend = TRUE)
# ) %>%
#   add_glassbrain(hemisphere = c("left", "right"), 
#                  colour = "#cecece", opacity = 0.25)

p <- plotly::layout(p,
                    scene = list(
                      camera = list(
                        eye = list(x = -20, y = 20, z = -10)  # 负 x → 左侧, 负 y → 外侧, 负 z → 下部
                      ),
                      xaxis = list(visible = FALSE),
                      yaxis = list(visible = FALSE),
                      zaxis = list(visible = FALSE)
                    ),
                    hoverlabel = list(bgcolor = "black", font = list(size = 12))) # 鼠标提示风格
p

library(htmlwidgets)  # 用于保存 plotly / ggseg3d 图
# 保存为 HTML 文件
# saveWidget(p, file = ".../write_loading_back_25xin/PIC_brain_R/loading_allRegion/G1_subcor18_3d.html", selfcontained = TRUE)
# # 手动export保存
# ==========================loading view END =========================
