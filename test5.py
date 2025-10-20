import numpy as np
from skimage.morphology import skeletonize, thin
from skimage import data
import matplotlib.pyplot as plt
from skimage.util import invert
import toolkit_3D as tk3

img_dict = tk3.get_nii("data_p2.3_preprocess/1_pre.nii.gz")
img = img_dict['artery']
img_info = img_dict['info']

print(np.sum(img))
# perform skeletonization
img = skeletonize(img)
print(np.unique(img))
print(np.sum(img))
print(img.shape)

img = tk3.image_dilation(img, 2)
tk3.save_nii(img, "test5.nii.gz", img_info)

# # display results
# fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8, 4), sharex=True, sharey=True)
#
# ax = axes.ravel()