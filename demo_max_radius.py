import logging
import os
import xlwt

import toolkit_3D as tk3
import toolkit_main as tkm

# region <------------------------- SET PARAMETERS ------------------------->
dataset_list_path = "data_list(0630new).txt"
result_xls_path = "results/result_tumor.xls"
detect_artery = True
detect_vein = True
# endregion <------------------------- SET PARAMETERS ------------------------->

if not os.path.exists(result_xls_path):
    with open(result_xls_path, mode='w', encoding='utf-8') as ff:
        logging.info('"' + result_xls_path + '" does not exist, successfully created.')

book = xlwt.Workbook()
tumor_sheet = book.add_sheet('Contact 2D Analysis')
title = ['Case ID',  # 0
         'Tumor Volume',  # 1
         'Tumor Diameter',  # 2
         'Tumor Diameter Position',  # 3
         ]
for col, t in enumerate(title):
    tumor_sheet.write(0, col, t)

# Image files loading
file_list = tkm.get_img_file_list(dataset_list_path)

vein_empty_list = []

row = 1
for file_dict in file_list:
    img_dict = tk3.get_nii(file_dict["img_path"])
    img_tumor = img_dict["tumor"]
    img_info = img_dict["info"]

    spacing = img_info[1]

    spacing = (spacing[2], spacing[1], spacing[0])
    print(spacing)

    shape = img_tumor.shape

    tumor_dict = tkm.get_tumor_info(img_tumor, spacing)
    tumor_sheet.write(row, 0, file_dict["img_id"])
    tumor_sheet.write(row, 1, int(tumor_dict["volume"]))
    tumor_sheet.write(row, 2, float(tumor_dict["diameter"]))
    tumor_sheet.write(row, 3, str(tumor_dict["diameter_voxels"]))
    row += 1

book.save(result_xls_path)
