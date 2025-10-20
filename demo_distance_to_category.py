import cc3d
import numpy as np
from scipy import ndimage
import SimpleITK as sitk
from skimage import morphology

import toolkit_3D as tk3
import SkeletonAnalysis as skele
from graph_v7 import get_skeleton_summary


img_dict = tk3.get_nii("data/39_seg.nii.gz")
img = img_dict["vein"]
img_info = img_dict["info"]
shape = img.shape

_, path_list = get_skeleton_summary(img, print_info=False)
path_list = skele.remove_small_skeleton(img, path_list, length_threshold=3)
path_list = skele.get_path_weight(img, path_list)

new_img = skele.distance_to_category(path_list, img, weight_decay=1)

tk3.save_nii(new_img, "distance_to_category0.nii.gz", img_info)
