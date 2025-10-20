import numpy as np
import xlwt
from scipy import ndimage
import toolkit_3D as tk3
import tookit_main as tkm

# region <------------------------- SET PARAMETERS ------------------------->
filter_artery = True
filter_vein = True
dataset_path = "data_1_preprocess"
result_xls_path = "results/result_ellipticity.xls"
brief_path = "resources/brief_PDAC.xlsx"

interest_dist = 10
# endregion <------------------------- SET PARAMETERS ------------------------->

file_list = tkm.get_img_file_list(dataset_path)

# region <====== Excel Initialization ======>

# New Excel
book = xlwt.Workbook()
# Initialize 2 sheets
ellipticity_sheet = book.add_sheet('Ellipticity')

# Inject titles
ellipticity_title = ['Case ID',  # 0
                     'Target',  # 1
                     'Slice',  # 2
                     'Position',  # 3
                     'Ellipticity',  # 4
                     ]
for col, t in enumerate(ellipticity_title):
    ellipticity_sheet.write(0, col, t)

style_red = xlwt.easyxf('pattern: pattern solid, fore_colour red;')

row = 1
# endregion <====== Excel Initialization ======>


for file_dict in file_list:

    img_dict = tk3.get_nii(file_dict["img_path"])
    img_tumor = img_dict["tumor"]
    img_info = img_dict["info"]
    shape = img_tumor.shape

    for target in ["artery", "vein"]:
        img = img_dict[target]
        dist_map = np.where(img_tumor == 1, 0, 1)
        dist_map = ndimage.distance_transform_edt(dist_map)
        dist_map = np.multiply(dist_map, img)
        dist_map = np.where(dist_map < interest_dist, 1, 0)

        for slice_num in range(0, shape[0]):
            slice = img[slice_num, :, :]
            dist_map_slice = dist_map[slice_num, :, :]
            isl_num, isl_maps = tk3.get_islands_num(slice)

            for isl_id in range(1, isl_num + 1):
                isl_map = np.where(isl_maps == isl_id, 1, 0)
                interest_map = np.multiply(isl_map, dist_map_slice)
                if np.sum(interest_map[interest_map == 1]) > 0:
                    ellipticity = tk3.get_ellipticity(isl_map)
                    isl_size = np.sum(isl_map)
                    isl_positions = np.where(isl_map == 1)
                    isl_position = (isl_positions[0][0], isl_positions[1][0])

                    # Case ID
                    ellipticity_sheet.write(row, 0, int(file_dict["img_id"]))
                    # Target
                    ellipticity_sheet.write(row, 1, target)
                    # Slice
                    ellipticity_sheet.write(row, 2, slice_num + 1)
                    # Position
                    ellipticity_sheet.write(row, 3, str(isl_position))
                    # Ellipticity
                    ellipticity_sheet.write(row, 4, float(ellipticity))
                    row += 1

# Save result xls
book.save(result_xls_path)
