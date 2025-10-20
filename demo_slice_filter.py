# import numpy as np
# import tqdm
# import xlwt
# from scipy import ndimage
#
# import toolkit_3D as tk3
# import tookit_main as tkm
# import SkeletonAnalysis as skele
# from graph_v7 import get_skeleton_summary
#
# # region <------------------------- SET PARAMETERS ------------------------->
# filter_artery = True
# filter_vein = True
# dataset_path = "data"
# result_xls_path = "results/result_thin_details.xls"
# brief_path = "resources/brief_PDAC.xlsx"
# # endregion <------------------------- SET PARAMETERS ------------------------->
#
# # Image files loading
# file_list = tkm.get_img_file_list(dataset_path)
#
# for file_dict in file_list:
#     img_dict = tk3.get_nii(file_dict["img_path"])
#     img_tumor = img_dict["tumor"]
#     img_info = img_dict["info"]
#     shape = img_tumor.shape
#     img_artery = None
#     img_vein = None
#
#     # region <====== Artery ======>
#     if filter_artery:
#         img = img_dict["artery"]
#         for slice_num in range(0, shape[0]):
#             slice = img[slice_num, :, :]
#             slice = tk3.remove_islands(slice, threshold_size=10)
#             img[slice_num, :, :] = slice
#         img = tk3.remove_islands(img, threshold_size=-1)
#
#         for slice_num in range(0, shape[1]):
#             slice = img[:, slice_num, :]
#             slice = tk3.remove_islands(slice, threshold_size=5)
#             img[:, slice_num, :] = slice
#         img = tk3.remove_islands(img, threshold_size=-1)
#
#         for slice_num in range(0, shape[2]):
#             slice = img[:, :, slice_num]
#             slice = tk3.remove_islands(slice, threshold_size=5)
#             img[:, :, slice_num] = slice
#         img = tk3.remove_islands(img, threshold_size=-1)
#         img_artery = img.copy()
#     else:
#         img_artery = img_dict["artery"]
#     # endregion <====== Artery ======>
#
#     # region <====== Vein ======>
#     if filter_vein:
#         img = img_dict["vein"]
#         for slice_num in range(0, shape[0]):
#             slice = img[slice_num, :, :]
#             slice = tk3.remove_islands(slice, threshold_size=10)
#             img[slice_num, :, :] = slice
#         img = tk3.remove_islands(img, threshold_size=-1)
#
#         for slice_num in range(0, shape[1]):
#             slice = img[:, slice_num, :]
#             slice = tk3.remove_islands(slice, threshold_size=5)
#             img[:, slice_num, :] = slice
#         img = tk3.remove_islands(img, threshold_size=-1)
#
#         for slice_num in range(0, shape[2]):
#             slice = img[:, :, slice_num]
#             slice = tk3.remove_islands(slice, threshold_size=5)
#             img[:, :, slice_num] = slice
#         img = tk3.remove_islands(img, threshold_size=3000)
#         dist_map = ndimage.distance_transform_edt(np.where(img == 1, 0, 1))
#         dist_map = np.multiply(dist_map, img_dict["vein"])
#         dist_map = np.where(dist_map == 0, 1000, dist_map)
#         compensate = np.where(dist_map <= 5, 1, 0)
#         img += compensate
#         img = tk3.remove_islands(img, threshold_size=-1)
#         img_vein = img.copy()
#     else:
#         img_vein = img_dict["vein"]
#     # endregion <====== Vein ======>
#
#     tk3.save_nii(img_artery + img_tumor * 2 + img_vein * 3, "data_slice_filter/" + file_dict["img_id"] + "_sf.nii.gz", img_info)

import toolkit_3D as tk3
import tookit_main as tkm

# region <------------------------- SET PARAMETERS ------------------------->
filter_artery = True
filter_vein = True
# dataset_path = "newdata"
dataset_path = "newdata_temp"
result_xls_path = "results/result_thin_details.xls"
brief_path = "resources/brief_PDAC.xlsx"
# endregion <------------------------- SET PARAMETERS ------------------------->

# Image files loading
file_list = tkm.get_img_file_list(dataset_path)

for file_dict in file_list:
    img_dict = tk3.get_nii(file_dict["img_path"])
    img_tumor = img_dict["tumor"]
    img_pancreas = img_dict["pancreas"]
    img_duct = img_dict["duct"]
    img_info = img_dict["info"]
    shape = img_tumor.shape
    print(shape)

    # region <====== Artery ======>
    if filter_artery:
        keep_list = tkm.get_keep_list(file_dict["img_id"], "artery")
        print(file_dict["img_id"])
        suspect_size = tkm.get_suspect_size(file_dict["img_id"], "artery", default_suspect_size=1000)
        img_artery = tkm.slice_filter(img_dict["artery"],
                                      threshold_size_tuple=(15, 8, 8),
                                      suspect_size=suspect_size,
                                      compensate_dist=0,
                                      keep_list=keep_list
                                      )
        # img = img_dict["artery"]
        # for slice_num in range(0, shape[0]):
        #     slice = img[slice_num, :, :]
        #     slice = tk3.remove_islands(slice, threshold_size=10)
        #     img[slice_num, :, :] = slice
        # img = tk3.remove_islands(img, threshold_size=-1)
        #
        # for slice_num in range(0, shape[1]):
        #     slice = img[:, slice_num, :]
        #     slice = tk3.remove_islands(slice, threshold_size=5)
        #     img[:, slice_num, :] = slice
        # img = tk3.remove_islands(img, threshold_size=-1)
        #
        # for slice_num in range(0, shape[2]):
        #     slice = img[:, :, slice_num]
        #     slice = tk3.remove_islands(slice, threshold_size=5)
        #     img[:, :, slice_num] = slice
        # img = tk3.remove_islands(img, threshold_size=-1)
        # img_artery = img.copy()
    else:
        img_artery = img_dict["artery"]
    # endregion <====== Artery ======>

    # region <====== Vein ======>
    if filter_vein:
        suspect_size = tkm.get_suspect_size(file_dict["img_id"], "vein", default_suspect_size=3000)
        img_vein = tkm.slice_filter(img_dict["vein"],
                                    threshold_size_tuple=(10, 5, 5),
                                    # suspect_size=3000,
                                    suspect_size=suspect_size,
                                    compensate_dist=5
                                    )
    else:
        img_vein = img_dict["vein"]
    # endregion <====== Vein ======>

    tk3.save_nii(img_artery + img_tumor * 2 + img_vein * 3 + img_pancreas * 4 + img_duct * 5, "newdata_0_slice_filter/" + file_dict["img_id"] + "_sf.nii.gz",
                 img_info)
