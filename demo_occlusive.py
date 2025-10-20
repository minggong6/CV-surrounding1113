import tqdm
import xlwt

import toolkit_3D as tk3
import tookit_main as tkm
import SkeletonAnalysis as skele
from graph_v7 import get_skeleton_summary

# region <------------------------- SET PARAMETERS ------------------------->
detect_artery = False
detect_vein = True
# dataset_path = "data_1_preprocess"
dataset_path = "data"
result_xls_path = "results/result_thin_details.xls"
brief_path = "resources/brief_PDAC.xlsx"
# endregion <------------------------- SET PARAMETERS ------------------------->

# Image files loading
file_list = tkm.get_img_file_list(dataset_path)

brief_list = tkm.get_brief(brief_path)
occlusive_vein_list = []
# # region <====== Excel Initialization ======>
#
# # New Excel
# book = xlwt.Workbook()
# # Initialize 2 sheets
# sheet = book.add_sheet('sheet1')
#
# # Inject titles
# title = ['Case ID',  # 0
#          'Case Status',  # 1
#          'Target',  # 2
#          'Part ID',  # 3
#          'Part Radis',  # 4
#          'Part Thin',  # 5
#          'Ratio',  # 6
#          ]
#
# for col, t in enumerate(title):
#     sheet.write(0, col, t)
#
# style_red = xlwt.easyxf('pattern: pattern solid, fore_colour red;')
# row = 1
# # endregion <====== Excel Initialization ======>


for file_dict in file_list:

    img_dict = tk3.get_nii(file_dict["img_path"])
    img_tumor = img_dict["tumor"]
    img_info = img_dict["info"]
    shape = img_tumor.shape



    # region <====== Strategy of Vein ======>
    if detect_vein:
        target = "vein"
        img = img_dict[target]
        isl_num, isl_size_list, isl_img, total_size = tk3.get_island_info(img)

        for i in range(0, isl_num):
            isl_size_list[i] = int(isl_size_list[i] / total_size * 100)

        print("Image " + file_dict["img_id"] + " - " + target + ": ")
        print("    " + str(isl_num) + " islands: ", end="")
        for isl_size in isl_size_list:
            print(str(isl_size) + "%, ", end="")
        print()
        if isl_size_list[1] > 15:
            occlusive_vein_list.append(int(file_dict["img_id"]))

            # for thin_dict in thin_list:
            #     # Case ID
            #     sheet.write(row, 0, int(file_dict["img_id"]))
            #     # Case Status
            #     sheet.write(row, 1, deformation)
            #     # Target
            #     sheet.write(row, 2, target)
            #     # Part ID
            #     sheet.write(row, 3, int(thin_dict["part_id"]))
            #     # Part Radis
            #     sheet.write(row, 4, float(thin_dict["part_radis"]))
            #     # Part Thin
            #     sheet.write(row, 5, float(thin_dict["thin_degree"]))
            #     # Ratio
            #     sheet.write(row, 6, float(thin_dict["thin_degree"] / thin_dict["part_radis"]))
            #     row += 1
    # endregion <====== Strategy of Vein ======>

    # region <====== Strategy of Artery ======>
    if detect_artery:
        target = "artery"
        img = img_dict[target]

    # endregion <====== Strategy of Artery ======>

# book.save(result_xls_path)
occlusive_vein_list.sort()
print(occlusive_vein_list)