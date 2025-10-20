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
# data_list_path = "data_list(all_wy_processed)1.txt"
data_list_path = "data_list(final87).txt"

timer = time.strftime("-%Y%m%d-%H%M%S", time.localtime())
result_xls_path = "results/result_thin" + timer + "_full_info(test).xls"
thin_GT_path = "resources/thin_GT.xlsx"
# endregion <------------------------- SET PARAMETERS ------------------------->

# Image files loading
file_list = tkm.get_img_file_list(data_list_path)

# img_id = '27'
# file_dict = {
#     "img_id": img_id,
#     "img_path": 'data_p2.3_preprocess\\' + img_id + '_pre.nii.gz',
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
title1 = ['',  # 0
          'Artery',  # 1
          '',  # 2
          '',  # 3
          'Vein',  # 4
          ]
title2 = ['Case ID',  # 0
          'Coordinate Info',  # 1
          'Pred',  # 2
          'GT',  # 3
          'Coordinate Info',  # 4
          'Pred',  # 5
          'GT',  # 6
          'Artery Status',  # 7
          'Vein Status',  # 8
          ]

row = 0
for col, t in enumerate(title1):
    sheet.write(row, col, t)
row += 1
for col, t in enumerate(title2):
    sheet.write(row, col, t)
row += 1
# endregion <====== Excel Initialization ======>


for file_dict in file_list:

    # if file_dict["img_id"] not in ['5', '7', '15', '25', '30', '31', '32', '35', '58', '59']:
    #     continue

    # if file_dict["img_id"] != '31':
    #     continue

    img_dict = tk3.get_nii(file_dict["img_path"])
    img_tumor = img_dict["tumor"]
    img_info = img_dict["info"]
    shape = img_tumor.shape

    tumor_dist_graph = np.where(img_tumor > 0, 0, 1)
    tumor_dist_graph = ndimage.distance_transform_edt(tumor_dist_graph)
    thin_graph = np.zeros(shape)

    # Case ID
    sheet.write(row, 0, int(file_dict["img_id"]))

    tm = np.zeros(shape)

    # region <====== Strategy of Artery ======>
    if detect_artery:
        target = "artery"
        img = img_dict[target]
        print("Image " + file_dict["img_id"] + " - " + target + " processing: ")

        # skeleton = tks.Skeleton(img, img_tumor, thin_degree=0.33, max_vessel_tumor_dist=2, max_min_radis_dist=1)
        # skeleton = tks.Skeleton(img, img_tumor, thin_degree=0.5, max_vessel_tumor_dist=1, max_min_radis_dist=1)
        # skeleton = tks.Skeleton(img, img_tumor, thin_degree=0.3, max_vessel_tumor_dist=1.5, max_min_radis_dist=2, related_range_bias=0.1)
        skeleton = tks.Skeleton(img, img_tumor, thin_degree=0.3, max_vessel_tumor_dist=1.5, max_min_radis_dist=1,
                                related_range_bias=1)
        skeleton.process_thin_analysis()
        not_found = True
        target_info = ''
        for thin_point in skeleton.thin_point_list:
            thin_graph[thin_point.coordinate] = 7
            target_info += str(thin_point)
            if not_found:
                not_found = False
        # Coordinate Info
        sheet.write(row, 1, target_info)
        # Pred
        sheet.write(row, 2, 0 if target_info == '' else 1)

    # endregion <====== Strategy of Artery ======>

    # region <====== Strategy of Vein ======>
    if detect_vein:
        target = "vein"
        img = img_dict[target]
        print("Image " + file_dict["img_id"] + " - " + target + " processing: ")

        # skeleton = tks.Skeleton(img, img_tumor, thin_degree=0.33, max_vessel_tumor_dist=2, max_min_radis_dist=1)
        skeleton = tks.Skeleton(img, img_tumor, thin_degree=0.5, max_vessel_tumor_dist=1.5, max_min_radis_dist=1)
        skeleton.process_thin_analysis()
        not_found = True
        target_info = ''
        for thin_point in skeleton.thin_point_list:
            thin_graph[thin_point.coordinate] = 8
            target_info += str(thin_point)
            if not_found:
                not_found = False
        # Coordinate Info
        sheet.write(row, 4, target_info)
        # Pred
        sheet.write(row, 5, 0 if target_info == "" else 1)

    # endregion <====== Strategy of Vein ======>
    kernel = morphology.ball(3)
    img_dilation = morphology.dilation(thin_graph, kernel)
    origin_img = img_dict['origin']
    for vox in tk3.tuple_to_list(np.where(img_dilation == 7)):
        origin_img[vox] = 7
    for vox in tk3.tuple_to_list(np.where(img_dilation == 8)):
        origin_img[vox] = 8
    tk3.save_nii(origin_img, 'data_p2.2_thinmark3/' + file_dict['img_id'] + '_tm.nii.gz', img_dict['info'])

    row += 1

book.save(result_xls_path)
