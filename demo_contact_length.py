import os

import numpy as np
import tqdm
import xlwt
from scipy import ndimage

import toolkit_3D as tk3
import toolkit_main as tkm
import toolkit_skeleton as tks

# region <------------------------- SET PARAMETERS ------------------------->
data_list_path = "data_list(0726).txt"
result_xls_path = "results/result_contact_length_0807.xls"
part_paths = {
    'artery': r"data_2_part_artery_remerge-0727",
    'vein': r"data_2_part_vein_remerge-0727"
}
into_part = True
# endregion <------------------------- SET PARAMETERS ------------------------->

# Image files loading
file_list = tkm.get_img_file_list(data_list_path)

# img_id = '28'
# file_dict = {
#     "img_id": img_id,
#     "img_path": 'data_p2.2_preprocess\\' + img_id + '_pre.nii.gz',
#     "img_contact_path": None
# }
# file_list = [file_dict, ]

book = xlwt.Workbook()
sheet_cl = book.add_sheet('Contact Length')

if into_part:
    title = ['Case ID',  # 0
             'CA',  # 1
             'CHA',  # 2
             'SMA',  # 3
             'PV',  # 4
             'SMV'  # 5
             ]
else:
    title = ['Case ID',  # 0
             'Artery Contact Length',  # 1
             'Vein Contact Length',  # 2
             ]

for col, t in enumerate(title):
    sheet_cl.write(0, col, t)

sheet_cv = book.add_sheet('Contact Voxels')

if into_part:
    title = ['Case ID',  # 0
             'CA',  # 1
             'CHA',  # 2
             'SMA',  # 3
             'PV',  # 4
             'SMV'  # 5
             ]
else:
    title = ['Case ID',  # 0
             'Artery Contact Length',  # 1
             'Vein Contact Length',  # 2
             ]

for col, t in enumerate(title):
    sheet_cv.write(0, col, t)

row = 1

for file_dict in file_list:

    img_dict = tk3.get_nii(file_dict["img_path"])
    img_tumor = img_dict["tumor"]
    case_id = file_dict["img_id"]

    img_info = img_dict["info"]
    shape = img_tumor.shape
    spacing = img_info[1]

    sp_z, sp_y, sp_x = spacing
    sp_xy = (sp_x ** 2 + sp_y ** 2) ** 0.5
    sp_yz = (sp_y ** 2 + sp_z ** 2) ** 0.5
    sp_xz = (sp_x ** 2 + sp_z ** 2) ** 0.5
    sp_xyz = (sp_x ** 2 + sp_y ** 2 + sp_z ** 2) ** 0.5

    posi_dist_list = [((1, 0, 0), sp_x), ((-1, 0, 0), sp_x),
                      ((0, 1, 0), sp_y), ((0, -1, 0), sp_y),
                      ((0, 0, 1), sp_z), ((0, 0, -1), sp_z),
                      ((1, 1, 0), sp_xy), ((1, -1, 0), sp_xy), ((-1, 1, 0), sp_xy), ((-1, -1, 0), sp_xy),
                      ((0, 1, 1), sp_yz), ((0, 1, -1), sp_yz), ((0, -1, 1), sp_yz), ((0, -1, -1), sp_yz),
                      ((1, 0, 1), sp_xz), ((1, 0, -1), sp_xz), ((-1, 0, 1), sp_xz), ((-1, 0, -1), sp_xz),
                      ((1, 1, 1), sp_xyz), ((-1, 1, 1), sp_xyz), ((1, -1, 1), sp_xyz), ((1, 1, -1), sp_xyz),
                      ((1, -1, -1), sp_xyz), ((-1, 1, -1), sp_xyz), ((-1, -1, 1), sp_xyz), ((-1, -1, -1), sp_xyz), ]

    sheet_cl.write(row, 0, case_id)
    save_images = {"artery": np.zeros(shape),
                   "vein": np.zeros(shape)}

    for target in ["artery", "vein"]:
        img_target = img_dict[target]
        img_part_path = os.path.join(part_paths[target], str(case_id) + '_' + target + '.nii.gz')
        if not os.path.exists(img_part_path):
            continue
        img_part = tk3.get_any_nii(img_part_path)["img"]
        skeleton = tks.Skeleton(img_target, tumor_label=img_tumor)

        if into_part:
            part_length_dict, part_contact_dict = skeleton.process_contact_length(posi_dist_list, part_graph=img_part, target=target)
            if target == "artery":
                sheet_cl.write(row, 1, part_length_dict['2'])
                sheet_cl.write(row, 2, part_length_dict['3'])
                sheet_cl.write(row, 3, part_length_dict['6'])
                sheet_cv.write(row, 1, part_contact_dict['2'])
                sheet_cv.write(row, 2, part_contact_dict['3'])
                sheet_cv.write(row, 3, part_contact_dict['6'])
            elif target == "vein":
                sheet_cl.write(row, 4, part_length_dict['1'])
                sheet_cl.write(row, 5, part_length_dict['2'])
                sheet_cv.write(row, 4, part_contact_dict['1'])
                sheet_cv.write(row, 5, part_contact_dict['2'])
        else:
            contact_length, contact_skele_graph = skeleton.process_contact_length(posi_dist_list)
            # save_images[target] = contact_skele_graph

            if target == "artery":
                sheet_cl.write(row, 1, contact_length)
            elif target == "vein":
                sheet_cl.write(row, 2, contact_length)

            print("case " + str(file_dict["img_id"]) + " " + target + "  contact length = " + str(contact_length))
    # save_image = np.where(save_images["artery"] == 1, 1, 0) + np.where(save_images["artery"] == 2, 4, 0) \
    #              + np.where(save_images["vein"] == 1, 3, 0) + np.where(save_images["vein"] == 2, 5, 0)
    # save_image = tk3.image_dilation(save_image, 2)
    # save_image = np.multiply(np.where(img_tumor == 1, 0, 1), save_image) + img_tumor * 2
    # tk3.save_nii(save_image, os.path.join('data_contact_skeleton1', file_dict['img_id'] + "_cs.nii.gz"), img_dict['info'])
    row += 1
    # tk3.save_nii(test_path_graph, "test_graph.nii.gz", img_dict['info'])
    # tk3.save_nii(test_cpath_graph, "test_cgraph.nii.gz", img_dict['info'])
    # break
book.save(result_xls_path)