import os
import cv2
import numpy as np
import toolkit_3D as tk3
from toolkit_skeleton import Skeleton

preprocess_path = 'data_p2.4_preprocess'

img_id = '1'

preprocess_filename = os.path.join(preprocess_path, img_id + '_pre.nii.gz')

file_dict = {
    "img_id": img_id,
    "img_path": 'data_p2.3_preprocess\\' + img_id + '_pre.nii.gz',
    "img_contact_path": None
}

# img_id = '83'
# file_dict = {
#     "img_id": img_id,
#     "img_path": 'data_p2.3_preprocess\\' + img_id + '_pre.nii.gz',
#     "img_contact_path": None
# }

file_list = [file_dict, ]

img_dict = tk3.get_nii(preprocess_filename)
img_tumor = img_dict["tumor"]
img_info = img_dict["info"]
shape = img_tumor.shape

img = img_dict['artery']
skeleton = Skeleton(img)
skeleton.generate_skele_point_list()
skeleton.generate_radius_graph()
skeleton.generate_path_graph()
img = tk3.image_erosion(img, 1).astype('float64')
img += skeleton.radius_graph.astype('float64')
img = np.where(img > 0, 1, 0).astype('float64')
img = tk3.image_dilation(img, 1).astype('float64')
img = tk3.image_erosion(img, 1).astype('float64')
img += skeleton.radius_graph.astype('float64')
img = np.where(img > 0, 1, 0).astype('float64')
img = tk3.image_dilation(img, 1).astype('float64')

tk3.save_nii(img, 'test.nii.gz', img_info)