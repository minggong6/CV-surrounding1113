import logging
import os.path
from scipy import ndimage
import toolkit_main as tkm

import numpy as np
import toolkit_3D as tk3

img_path = r"data_2_part_vein/15_vein.nii.gz"
save_path = r"data_2_part_vein/15_vein1.nii.gz"


# img_path = r"data_p2.2_preprocess/" + img_id + r"_pre.nii.gz"
# save_path = r"data_2_part_vein/" + img_id + r"_vein.nii.gz"
img_dict = tk3.get_any_nii(img_path)

img = img_dict["img"]

# img_target = np.where(img == cut_target_id, 1, 0).astype(np.uint8)

for x in range(0, img.shape[0]):
    for y in range(0, img.shape[1]):
        for z in range(0, img.shape[2]):
            if z < 146 and img[(x, y, z)] == 1:
                img[(x, y, z)] = 7

# img_target = np.zeros_like(img_target) + np.where(img_target == 2, 1, 0) + np.where(img_target == 1, 2, 0) + np.where(img == 2, 3, 0)

tk3.save_nii(img, save_path, img_dict["info"])
