import os

import numpy as np

import toolkit_3D as tk3

preprocess_path = 'data_p2.4_preprocess'

img_id = '1'

preprocess_filename = os.path.join(preprocess_path, img_id + '_pre.nii.gz')

file_dict = {
    "img_id": img_id,
    "img_path": 'data_p2.4_preprocess\\' + img_id + '_pre.nii.gz',
    "img_contact_path": None
}
file_list = [file_dict, ]

img_dict = tk3.get_nii(preprocess_filename)
img_tumor = img_dict["tumor"]
img_info = img_dict["info"]
shape = img_tumor.shape

img = img_dict['artery']

visibility = np.zeros(shape)
vox1 = (40, 50, 90)
vox2 = (30, 80, 100)
n = max(abs(vox1[0] - vox2[0]), abs(vox1[1] - vox2[1]), abs(vox1[2] - vox2[2]))
space = ((vox1[0] - vox2[0]) / n, (vox1[1] - vox2[1]) / n, (vox1[2] - vox2[2]) / n)
for i in range(0, n):
    posi = (int(vox2[0] + i * space[0]), int(vox2[1] + i * space[1]), int(vox2[2] + i * space[2]))
    visibility[posi] = 1
# visibility = tk3.image_dilation(visibility, 2)
tk3.save_nii(visibility, 'test0.nii.gz', img_info)

# direction = (38 - 30, 88 - 76, 114 - 136)
# direction = (30 - 30, 50 - 80, 100 - 100)
direction = (vox1[0] - vox2[0], vox1[1] - vox2[1], vox1[2] - vox2[2])
direction_len = ((direction[0]) ** 2 + (direction[1]) ** 2 + (direction[2]) ** 2) ** 0.5
direction = (direction[0] / direction_len, direction[1] / direction_len, direction[2] / direction_len)
# rotated_img = tk3.image_rotation(visibility, (37, 87, 113), direction, rotation_range=0)
rotated_img = tk3.image_rotation(visibility, (30, 50, 100), direction, rotation_range=0)
save_img = rotated_img * 2 + visibility
save_img = tk3.image_dilation(save_img, 2)
tk3.save_nii(save_img, 'test.nii.gz', img_info)
