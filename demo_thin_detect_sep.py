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
data_list_path = "data_list(0726).txt"

timer = time.strftime("-%Y%m%d-%H%M%S", time.localtime())
# result_xls_path = "results/result_thin" + timer + ".xls"
result_xls_path = "results/result_thin_0731test.xls"
origin_img_list_path = "data_list(0726).txt"
remerged_artery_list_path = "data_list_artery_part.txt"
remerged_vein_list_path = "data_list_vein_part.txt"
thin_img_save_path = "data_p2.3_thinmark_sep"
# endregion <------------------------- SET PARAMETERS ------------------------->

part_value_dict = {
    'CA': 2,
    'CHA': 3,
    'SMA': 6,
    'PV': 1,
    'SMV': 2
}

# Image files loading
origin_file_list = tkm.get_img_file_list(origin_img_list_path)
artery_file_list = tkm.get_img_file_list(remerged_artery_list_path)
vein_file_list = tkm.get_img_file_list(remerged_vein_list_path)
# img_id = '32'
# file_dict = {
#     "img_id": img_id,
#     "img_path": 'data_p2.2_preprocess\\' + img_id + '_pre1.nii.gz',
#     "img_contact_path": None
# }
# file_list = [file_dict, ]

assert len(artery_file_list) == len(artery_file_list) == len(vein_file_list), \
    "The case number of artery and vein do not match."

vessel_file_list = []
for origin_file_dict, artery_file_dict, vein_file_dict in zip(origin_file_list, artery_file_list, vein_file_list):
    assert origin_file_dict['img_id'] == artery_file_dict['img_id'] == vein_file_dict['img_id'], \
        f"Case id do not match: {artery_file_dict['img_id']}, {vein_file_dict['img_id']} "
    vessel_file_dict = {
        "case_id": artery_file_dict['img_id'],
        "origin_path": origin_file_dict['img_path'],
        "artery_path": artery_file_dict['img_path'],
        "vein_path": vein_file_dict['img_path'],
    }
    vessel_file_list.append(vessel_file_dict)

# region <====== Excel Initialization ======>
book = xlwt.Workbook()
sheet = book.add_sheet('sheet1')

style_red = xlwt.easyxf('pattern: pattern solid, fore_colour red;')
title_dict = {
    'Case ID': 0,  # 0
    'CA Info': 1,  # 1
    'CA Pred': 2,  # 2
    'CHA Info': 3,  # 3
    'CHA Pred': 4,  # 4
    'SMA Info': 5,  # 5
    'SMA Pred': 6,  # 6
    'PV Info': 7,  # 7
    'PV Pred': 8,  # 8
    'SMV Info': 9,  # 9
    'SMV Pred': 10,  # 10
}
save_dict = {
    'CA': 1,
    'CHA': 3,
    'SMA': 5,
    'PV': 7,
    'SMV': 9
}

row = 0
for title in title_dict.keys():
    col = title_dict[title]
    sheet.write(row, col, title)
row += 1
# endregion <====== Excel Initialization ======>

# case_list = [1, 2, 3, 4, 5, 7, 9, 10]
case_list = []

for file_dict in vessel_file_list:

    if len(case_list) > 0:
        if int(file_dict["case_id"]) not in case_list:
            continue

    origin_img_dict = tk3.get_nii(file_dict['origin_path'])
    artery_img_dict = tk3.get_any_nii(file_dict['artery_path'])
    vein_img_dict = tk3.get_any_nii(file_dict['vein_path'])

    part_img_dict = {
        'CA': np.where(artery_img_dict['img'] == part_value_dict['CA'], 1, 0),
        'CHA': np.where(artery_img_dict['img'] == part_value_dict['CHA'], 1, 0),
        'SMA': np.where(artery_img_dict['img'] == part_value_dict['SMA'], 1, 0),
        'PV': np.where(vein_img_dict['img'] == part_value_dict['PV'], 1, 0),
        'SMV': np.where(vein_img_dict['img'] == part_value_dict['SMV'], 1, 0)
    }
    img_tumor = origin_img_dict['tumor']
    img_info = origin_img_dict['info']
    shape = img_tumor.shape

    tumor_dist_graph = np.where(img_tumor > 0, 0, 1)
    tumor_dist_graph = ndimage.distance_transform_edt(tumor_dist_graph)


    # Case ID
    sheet.write(row, 0, file_dict["case_id"])

    overall_thin_graph = np.zeros(shape)
    for target in part_img_dict.keys():
        overall_thin_graph += part_img_dict[target] * save_dict[target]
    thin_graph = np.zeros(shape)

    for target in part_img_dict.keys():
        tumor_dist_threshold = 1
        max_min_radis_dist = 1
        img = part_img_dict[target]
        if np.sum(img) <= 0:
            continue

        print("Image " + file_dict["case_id"] + " - " + target + " processing: ")

        skeleton = tks.Skeleton(img, img_tumor)
        skeleton.process_thin_analysis()
        not_found = True
        coordinate_info = ""
        for point in skeleton.thin_point_list:
            thin_graph[point.coordinate] = save_dict[target] + 1
            point_info = '[path ' + str(point.path_id) + ': ' + str(point.coordinate) + ' - ' + str(
                point.radius) + '] '
            coordinate_info += point_info
            coordinate_info += '\n'
            if not_found:
                not_found = False
        # Coordinate Info
        sheet.write(row, title_dict[target + ' Info'], coordinate_info)
        # Pred
        sheet.write(row, title_dict[target + ' Pred'], 0 if coordinate_info == "" else 1)

    thin_graph = tk3.image_dilation(thin_graph, 2)
    print(np.unique(thin_graph))
    overall_thin_graph = np.multiply(np.where(thin_graph > 0, 0, 1), overall_thin_graph) + thin_graph
    print(np.unique(overall_thin_graph))
    overall_thin_graph = np.multiply(np.where(overall_thin_graph > 0, 0, 1), img_tumor) * 11 + overall_thin_graph
    print(np.unique(overall_thin_graph))
    tk3.save_nii(overall_thin_graph, os.path.join(thin_img_save_path, file_dict['case_id'] + '_tm.nii.gz'), img_info)
    row += 1
    # book.save(result_xls_path)
    # exit(10086)

book.save(result_xls_path)
