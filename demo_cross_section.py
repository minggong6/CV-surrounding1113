from matplotlib import pyplot as plt
from scipy import ndimage
from scipy.ndimage import rotate
import numpy as np
import os
import toolkit_3D as tk3
from toolkit_skeleton import Skeleton


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

max_path_tumor_dist = 10

skeleton = Skeleton(img)
skeleton.process_base()
# print(np.unique(skeleton.path_graph))
# print(skeleton.ordered_path_point_list_dict.keys())
tumor_dist_graph = ndimage.distance_transform_edt(np.where(img_tumor > 0, 0, 1))
path_mask = np.multiply(np.where(skeleton.path_graph > 0, 1, 0),
                        np.where(skeleton.path_graph < skeleton.cross_marker, 1, 0))
path_tumor_dist_graph = np.multiply(path_mask, tumor_dist_graph)

related_path_graph = np.multiply(np.where(path_tumor_dist_graph > 0, 1, 0),
                                 np.where(path_tumor_dist_graph < max_path_tumor_dist, 1, 0))

path_vox_list = tk3.tuple_to_list(np.where(related_path_graph > 0))
path_direction_list = []
for p_vox in path_vox_list:
    path_direction_list.append(skeleton.get_point_direction(p_vox))



for idx in range(0, len(path_vox_list)):
    rotated_img = tk3.image_rotation(img, path_vox_list[idx], path_direction_list[idx], rotation_range=10)
    cro_sec_matrix = rotated_img[path_vox_list[idx][0], :, :]
    ax = plt.matshow(cro_sec_matrix)
    plt.xlabel("$L$-Lagged Vectors")
    plt.ylabel("$K$-Lagged Vectors")
    plt.colorbar(ax.colorbar, fraction=0.025)
    ax.colorbar.set_label("$F(t)$")
    plt.title(f"The Cross Section of {path_vox_list[idx]}")
    plt.show()
    input()