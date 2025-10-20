import os
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


rotated_img = tk3.image_rotation(img, (30, 82, 58), (1, 0, 0), rotation_range=0)
tk3.save_nii(rotated_img, 'test.nii.gz', img_info)