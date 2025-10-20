import toolkit_3D as tk3
import tookit_main as tkm


# region <------------------------- SET PARAMETERS ------------------------->
fill_artery = True
fill_vein = True
dataset_path = "data"
result_xls_path = "results/result_thin_details.xls"
brief_path = "resources/brief_PDAC.xlsx"
# endregion <------------------------- SET PARAMETERS ------------------------->

# Image files loading
file_list = tkm.get_img_file_list(dataset_path)

for file_dict in file_list:
    img_dict = tk3.get_nii(file_dict["img_path"])
    img_tumor = img_dict["tumor"]
    img_info = img_dict["info"]
    shape = img_tumor.shape

    # region <====== Artery ======>
    if fill_artery:
        img_artery = tk3.fill_hole(img_dict["artery"])
    else:
        img_artery = img_dict["artery"]
    # endregion <====== Artery ======>

    # region <====== Vein ======>
    if fill_vein:
        img_vein = tk3.fill_hole(img_dict["vein"])
    else:
        img_vein = img_dict["vein"]
    # endregion <====== Vein ======>

    tk3.save_nii(img_artery + img_tumor * 2 + img_vein * 3, "data_fill_hole/" + file_dict["img_id"] + "_fh.nii.gz",
                 img_info)
