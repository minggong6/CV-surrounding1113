import logging
import os.path
from scipy import ndimage
import toolkit_main as tkm

import numpy as np
import toolkit_3D as tk3

img_path = r"data_p2.3_preprocess/14_pre.nii.gz"
save_path = r"data_2_part_vein_remerge-0727/14_vein_.nii.gz"
img_dict = tk3.get_any_nii(img_path)

img = img_dict["img"]

img = np.where(img == 3, 1, 0)

max_id = int(np.max(img))

new_img = np.zeros_like(img)

for origin_id in range(1, max_id + 1):

    img_target = np.where(img == origin_id, 1, 0).astype(np.uint8)

    isl_num, isl_size_list, isl_img, total_size = tk3.get_island_info(img_target)

    bias = int(np.max(new_img))

    isl_img = np.where(isl_img > 0, isl_img + bias, 0)

    new_img += isl_img

tk3.save_nii(new_img, save_path, img_dict["info"])