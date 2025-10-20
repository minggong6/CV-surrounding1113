import logging
import os

import cc3d
import numpy as np
import xlrd2
from scipy import ndimage
from skimage import morphology
import tqdm

import toolkit_3D as tk3
from data_regulator import read_nii_list
from toolkit_skeleton import Skeleton    

image_dict = tk3.get_any_nii(r'data_contact_skeleton\2_cs.nii.gz')
image = image_dict['img']
image_info = image_dict['info']

skeleton = np.where(image == 1, 1, 0) + np.where(image == 4, 1, 0) + np.where(image == 3, 3, 0) + np.where(image == 5, 3, 0)

tk3.save_nii(skeleton, "skeleton.nii.gz", image_info)