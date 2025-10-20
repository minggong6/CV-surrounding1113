import tqdm

import toolkit_3D as tk3
import toolkit_main as tkm
import SkeletonAnalysis as skele
from graph_v7 import get_skeleton_summary

# region <------------------------- SET PARAMETERS ------------------------->
dataset_path = "data"
detect_artery = True
detect_vein = True
# endregion <------------------------- SET PARAMETERS ------------------------->

# Image files loading
file_list = tkm.get_img_file_list(dataset_path)

for file_dict in file_list:

    img_dict = tk3.get_nii(file_dict["img_path"])
    img_tumor = img_dict["tumor"]
    img_info = img_dict["info"]
    shape = img_tumor.shape

    if detect_vein:
        # Detect the connection, and only keep the largest
        img_vein = tk3.remove_islands(img_dict["vein"], threshold_size=-1)
    else:
        img_vein = img_dict["vein"]

    if detect_vein:
        # Detect the connection, and only keep the largest
        img_artery = tk3.remove_islands(img_dict["artery"], threshold_size=-1)
    else:
        img_artery = img_dict["artery"]

    new_data = img_artery + img_tumor * 2 + img_vein * 3
    tk3.save_nii(new_data, "data_isl/" + file_dict["img_id"] + "_isl.nii.gz", img_info)
