import toolkit_3D as tk3
import toolkit_main as tkm

# region <------------------------- SET PARAMETERS ------------------------->
dataset_path = "data_p2.3"
# endregion <------------------------- SET PARAMETERS ------------------------->

# Image files loading
# file_list = tkm.get_img_file_list_old(dataset_path)


img_id = '72'
file_dict = {
    "img_id": img_id,
    "img_path": 'data_p2.3\\' + img_id + '_seg.nii.gz',
    "img_contact_path": None
}
file_list = [file_dict, ]

for file_dict in file_list:
    print(file_dict["img_path"])
    img_dict = tk3.get_nii(file_dict["img_path"])
    img_tumor = img_dict["tumor"]
    img_pancreas = img_dict["pancreas"]
    img_duct = img_dict["duct"]
    img_info = img_dict["info"]
    shape = img_tumor.shape

    # region <====== Artery ======>
    img_artery = img_dict["artery"]
    img_artery = tk3.remove_islands(img_artery, threshold_size=5000)
    img_artery = tk3.fill_hole(img_artery)
    # endregion <====== Artery ======>

    # region <====== Vein ======>
    img_vein = img_dict["vein"]
    img_vein = tk3.remove_islands(img_vein, threshold_size=5000)
    img_vein = tk3.fill_hole(img_vein)
    # endregion <====== Vein ======>

    tk3.save_nii(img_artery + img_tumor * 2 + img_vein * 3 + img_pancreas * 4 + img_duct * 5, "data_p2.3_preprocess/" + file_dict["img_id"] + "_pre.nii.gz",
                 img_info)
