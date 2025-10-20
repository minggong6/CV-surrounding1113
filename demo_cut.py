import logging
import os.path
from scipy import ndimage
import toolkit_main as tkm

import numpy as np
import toolkit_3D as tk3
# img_path = r"data_p2.3_preprocess/15_pre.nii.gz"
save_path = r"data_2_part_vein_remerge-0727/35_vein_.nii.gz"
# img_path = r"data_2_part_vein/" + img_id + r"_vein.nii.gz"
# save_path = r"data_2_part_vein/" + img_id + r"_vein_.nii.gz"


# img_path = r"data_p2.2_preprocess/" + img_id + r"_pre.nii.gz"
# save_path = r"data_2_part_vein/" + img_id + r"_vein.nii.gz"
img_path = r"data_2_part_vein_remerge-0727/35_vein.nii.gz"
img_dict = tk3.get_any_nii(img_path)
img = img_dict["img"]

# img_target = np.where(img > 0, 1, 0).astype(np.uint8)

range_center = (52 - 1, 110 - 1, 106 - 1)
range_radius = 50
range_mask = np.ones_like(img)
range_mask[range_center] = 0
range_mask = np.where(ndimage.distance_transform_edt(range_mask) < range_radius, 1, 0)

# 三个点的坐标
vox1 = (52 - 1, 110 - 1, 106 - 1)
vox2 = (56 - 1, 112 - 1, 116 - 1)
vox3 = (56 - 1, 115 - 1, 105 - 1)

plane3D = tk3.Plane3D(vox1, vox2, vox3)

for vox in tk3.tuple_to_list(np.where(range_mask > 0)):
    posi_encoding = plane3D.get_posi_encoding(vox)
    if img[vox] == 1:
        if posi_encoding > 0:
            img[vox] = 2



# img_target = np.zeros_like(img_target) + np.where(img_target == 2, 1, 0) + np.where(img_target == 1, 2, 0) + np.where(img == 2, 3, 0)

tk3.save_nii(img, save_path, img_dict["info"])
