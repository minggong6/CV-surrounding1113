import toolkit_3D as tk3
import toolkit_main as tkm
import logging
import numpy as np

img_path = r"data_2_part_vein/80_vein.nii.gz"
save_path = r"data_2_part_vein/80_vein__.nii.gz"
img_dict = tk3.get_any_nii(img_path)

anchor_point = (83, 111, 95)

img = img_dict["img"]

img_binary = np.where(img > 0, 1, 0)

isl_num, isl_size_list, isl_img, total_size = tk3.get_island_info(img_binary)

remove_value = isl_img[anchor_point]

remove_mask = np.where(isl_img == remove_value, 0, 1)

img = np.multiply(remove_mask, img)

tk3.save_nii(img, save_path, img_dict["info"])
