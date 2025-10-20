import os.path

import numpy as np
import tqdm
import xlwt
import time
from scipy import ndimage
from skimage import morphology

import toolkit_3D as tk3
import toolkit_main as tkm
import toolkit_skeleton as tks

# region <------------------------- SET PARAMETERS ------------------------->
detect_artery = True
detect_vein = True
# dataset_path = "data_1_preprocess"
# data_list_path = "data_list(all_wy_processed)1.txt"
data_list_path = "data_list(final87).txt"

timer = time.strftime("-%Y%m%d-%H%M%S", time.localtime())
result_xls_path = "results/result_thin" + timer + "(test).xls"
thin_GT_path = "resources/thin_GT.xlsx"
part_artery_path = "data_2_part_artery_remerge-0727"
part_vein_path = "data_2_part_vein_remerge-0727"
# endregion <------------------------- SET PARAMETERS ------------------------->

# Image files loading
file_list = tkm.get_img_file_list(data_list_path)
# file_list = file_list[33:]

# img_id = '5'
# file_dict = {
#     "img_id": img_id,
#     "img_path": 'data_p2.4_preprocess\\' + img_id + '_pre.nii.gz',
#     "img_contact_path": None
# }
# file_list = [file_dict, ]
# region <====== Excel Initialization ======>

# New Excel
book = xlwt.Workbook()
# Initialize 2 sheets
sheet = book.add_sheet('sheet1')

# Inject titles
style_red = xlwt.easyxf('pattern: pattern solid, fore_colour red;')
title = ['Case ID',  # 0
         'target',  # 1
         'radius_min',  # 2
         'radius_max',  # 3
         'radius_mean',  # 4
         'radius_ptp',  # 5
         'radius_var',  # 6
         'radius_std',  # 7
         'syn_rt_min',  # 8
         'syn_rt_max',  # 9
         'syn_rt_mean',  # 10
         'syn_rt_ptp',  # 11
         'syn_rt_var',  # 12
         'syn_rt_std',  # 13
         'mean_ratio',  # 14
         'ptp_ratio',  # 15
         'var_ratio',  # 16
         'std_ratio'  # 17
         ]

row = 0
for col, t in enumerate(title):
    sheet.write(row, col, t)
row += 1
# endregion <====== Excel Initialization ======>


for file_dict in file_list:

    img_dict = tk3.get_nii(file_dict["img_path"])
    img_tumor = img_dict["tumor"]
    img_info = img_dict["info"]
    shape = img_tumor.shape

    tumor_dist_graph = np.where(img_tumor > 0, 0, 1)
    tumor_dist_graph = ndimage.distance_transform_edt(tumor_dist_graph)
    thin_graph = np.zeros(shape)

    # region <====== Strategy of Vein ======>
    if detect_artery:
        target = "artery"
        img = img_dict[target]
        print("Image " + file_dict["img_id"] + " - " + target + " processing: ")

        # 想法：把 part_graph 读进来，在 path 阶段直接划分血管，把每一根血管中，与肿瘤相关的部分的 path 点全部放在一个 list 里，按照已有的分析法计算 4 个直接统计量和 4 个比值统计量
        part_img = tk3.get_any_nii(os.path.join(part_artery_path, file_dict["img_id"] + '_artery.nii.gz'))['img']
        skeleton = tks.Skeleton(img, img_tumor, thin_degree=0.5, max_vessel_tumor_dist=1.5, max_min_radis_dist=1,
                                related_range_bias=1, avg_radius=True, part_img=part_img, target=target)  # AVG ALL
        feature_dict = skeleton.process_thin_analysis()
        print(f'feature_dict: {feature_dict}')
        for vessel in ['CA', 'CHA', 'SMA']:
            sheet.write(row, 0, file_dict["img_id"])
            sheet.write(row, 1, vessel)
            if feature_dict[vessel] is None:
                # for i in range(2, len(title)):
                for i in range(2, 14):
                    sheet.write(row, i, -1)
            else:
                # for i in range(2, len(title)):
                for i in range(2, 14):
                    sheet.write(row, i, feature_dict[vessel][title[i]])
            row += 1

    # endregion <====== Strategy of Vein ======>

    # region <====== Strategy of Artery ======>
    if detect_vein:
        target = "vein"
        img = img_dict[target]
        print("Image " + file_dict["img_id"] + " - " + target + " processing: ")

        # 想法：把 part_graph 读进来，在 path 阶段直接划分血管，把每一根血管中，与肿瘤相关的部分的 path 点全部放在一个 list 里，按照已有的分析法计算 4 个直接统计量和 4 个比值统计量
        part_img = tk3.get_any_nii(os.path.join(part_vein_path, file_dict["img_id"] + '_vein.nii.gz'))['img']
        skeleton = tks.Skeleton(img, img_tumor, thin_degree=0.5, max_vessel_tumor_dist=1.5, max_min_radis_dist=1,
                                related_range_bias=1, avg_radius=True, part_img=part_img, target=target)  # AVG ALL
        feature_dict = skeleton.process_thin_analysis()
        print(f'feature_dict: {feature_dict}')
        for vessel in ['PV', 'SMV']:
            sheet.write(row, 0, file_dict["img_id"])
            sheet.write(row, 1, vessel)
            if feature_dict[vessel] is None:
                # for i in range(2, len(title)):
                for i in range(2, 14):
                    sheet.write(row, i, -1)
            else:
                # for i in range(2, len(title)):
                for i in range(2, 14):
                    sheet.write(row, i, feature_dict[vessel][title[i]])
            row += 1

    # endregion <====== Strategy of Artery ======>

book.save(result_xls_path)
