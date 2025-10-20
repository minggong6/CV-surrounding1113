import os

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
detect_vein = False
# dataset_path = "data_1_preprocess"
data_list_path = "data_list(all_wy_processed)1.txt"

timer = time.strftime("-%Y%m%d-%H%M%S", time.localtime())
result_xls_path = "results/result_part_thin" + timer + ".xls"
thin_GT_path = "resources/thin_GT.xlsx"

part_img_dir = 'data_2_part_artery_remerge'
# endregion <------------------------- SET PARAMETERS ------------------------->

# Image files loading
file_list = tkm.get_img_file_list(data_list_path)
max_id = 10

# region <====== Excel Initialization ======>

# New Excel
book = xlwt.Workbook()
# Initialize 2 sheets
sheet = book.add_sheet('sheet1')

# Inject titles
style_red = xlwt.easyxf('pattern: pattern solid, fore_colour red;')

title = ['Case ID',  # 0
         'Target',  # 1
         'Part ID',  # 2
         'Info',  # 3
         'Pred',  # 4
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
    img_id = file_dict["img_id"]
    shape = img_tumor.shape

    part_img_name = img_id + '_artery.nii.gz'
    part_img_path = os.path.join(part_img_dir, part_img_name)
    if not os.path.exists(part_img_path):
        continue
    part_img = tk3.get_any_nii(part_img_path)['img']

    # Case ID
    sheet.write(row, 0, file_dict["img_id"])

    # region <====== Strategy of Vein ======>
    if detect_artery:
        thin_graph = np.zeros(shape)
        for target_id in range(1, max_id + 1):
            target = "artery"
            img = np.where(part_img == target_id, 1, 0)
            if np.sum(img) <= 0:
                continue
            print("Image " + file_dict["img_id"] + " - " + target + " part " + str(target_id) + " processing: ")

            skeleton = tks.Skeleton(img, img_tumor)
            skeleton.process_thin_analysis()
            not_found = True
            coordinate_info = ""
            for point in skeleton.thin_point_list:
                thin_graph[point.coordinate] = target_id
                point_info = '[path ' + str(point.path_id) + ': ' + str(point.coordinate) + ' - ' + str(point.radius) + '] '
                coordinate_info += point_info
                coordinate_info += '\n'
                if not_found:
                    not_found = False

            print(coordinate_info)

            # Target
            sheet.write(row, 1, target)
            # Part ID
            sheet.write(row, 2, target_id)
            # Info
            sheet.write(row, 3, coordinate_info)
            # Pred
            sheet.write(row, 4, 0 if coordinate_info == "" else 1)

            row += 1

        kernel = morphology.ball(3)
        thin_graph = morphology.dilation(thin_graph, kernel)
        thin_graph = np.multiply(np.where(img_tumor > 0, 0, 1), thin_graph) + np.where(img_tumor > 0, max_id + 1, 0)

        tk3.save_nii(thin_graph, 'data_2_part_artery_thin_graph/' + file_dict['img_id'] + '_pat.nii.gz', img_dict['info'])



    # endregion <====== Strategy of Vein ======>

    # region <====== Strategy of Artery ======>
    if detect_vein:
        for target_id in (1, max_id + 1):
            target = "vein"
            tumor_dist_threshold = 1
            max_min_radis_dist = 1
            img = img_dict[target]
            print("Image " + file_dict["img_id"] + " - " + target + " processing: ")

            skeleton = tks.Skeleton(img, img_tumor)
            skeleton.process_thin_analysis()
            not_found = True
            coordinate_info = ""
            for point in skeleton.thin_point_list:
                thin_graph[point.coordinate] = 8
                point_info = '[path ' + str(point.path_id) + ': ' + str(point.coordinate) + ' - ' + str(
                    point.radius) + ']'
                coordinate_info += point_info
                coordinate_info += '\n'
                if not_found:
                    not_found = False
            # Coordinate Info
            sheet.write(row, 4, coordinate_info)
            # Pred
            sheet.write(row, 5, 0 if coordinate_info == "" else 1)
            # GT

    # endregion <====== Strategy of Artery ======>


book.save(result_xls_path)
