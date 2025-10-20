# import numpy as np
# import scipy.stats as sta
# import os
# import cv2
# import toolkit_3D as tk3
# from scipy import signal
# from toolkit_skeleton import Skeleton
#
#
# def gaussian_smooth_points(points, kernel_r, nsig=3):
#     """ 将 points 进行高斯平滑 """
#
#     smoothed_points = points.copy()
#     kernlen = kernel_r * 2 + 1
#     x = np.linspace(-nsig, nsig, kernlen + 1)
#     kern1d = np.diff(sta.norm.cdf(x))
#     kern1d = kern1d / kern1d.sum()
#
#     len_points = len(points)
#     for j in range(len_points):
#         if kernel_r < j < len_points - kernel_r:
#             sum_data = np.array([0.0, 0.0, 0.0], dtype=np.double)
#             for i in range(1, 2 * kernel_r + 1, 1):
#                 idx = j + i - kernel_r - 1
#                 sum_data += points[idx] * kern1d[i]
#             smoothed_points[j] = sum_data / (2 * kernel_r + 1)
#     return smoothed_points
#
#
# if __name__ == '__main__':
#     preprocess_path = 'data_p2.4_preprocess'
#
#     img_id = '1'
#
#     preprocess_filename = os.path.join(preprocess_path, img_id + '_pre.nii.gz')
#
#     file_dict = {
#         "img_id": img_id,
#         "img_path": 'data_p2.3_preprocess\\' + img_id + '_pre.nii.gz',
#         "img_contact_path": None
#     }
#     file_list = [file_dict, ]
#
#     img_dict = tk3.get_nii(preprocess_filename)
#     img_tumor = img_dict["tumor"]
#     img_info = img_dict["info"]
#     shape = img_tumor.shape
#
#     img = img_dict['artery']
#
#     # first build the smoothing kernel
#     sigma = 1.0  # width of kernel
#     x = np.arange(-3, 4, 1)  # coordinate arrays -- make sure they contain 0!
#     y = np.arange(-3, 4, 1)
#     z = np.arange(-3, 4, 1)
#     xx, yy, zz = np.meshgrid(x, y, z)
#     kernel = np.exp(-(xx ** 2 + yy ** 2 + zz ** 2) / (2 * sigma ** 2))
#
#     filtered = signal.convolve(img, kernel, mode="same")
#     print(f'range: {np.min(filtered)}, {np.max(filtered)}')
#     tk3.save_nii(np.where(filtered > 4, 1, 0), 'test.nii.gz', img_info)
#     # tk3.save_nii(filtered, 'test.nii.gz', img_info)
#
#
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator

vel = np.random.random((21, 30))
# grid old
x = np.arange(0, 21, 1)
y = np.arange(0, 30, 1)
grid_old = (x, y)
# grid new
# the limits of the interpolated x and y val have to be less than the original grid
x_new = np.arange(0.1, 19.9, 0.1)
y_new = np.arange(0.1, 28.9, 0.1)
grid_new = np.meshgrid(x_new, y_new)
grid_flattened = np.transpose(np.array([k.flatten() for k in grid_new]))
# Interpolation onto a finer grid
grid_interpol = RegularGridInterpolator(grid_old, vel, method='linear')
vel_interpol = grid_interpol(grid_flattened)
# Unflatten the interpolated velocities and store into a new variable.
index = 0
vel_new = np.zeros((len(x_new), len(y_new)))
for i in range(len(x_new)):
    for j in range(len(y_new)):
        vel_new[i, j] = vel_interpol[index]
        index += 1
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(grid_new[0], grid_new[1], vel_new.T, cmap="RdBu")
fig.set_size_inches(10, 10)
plt.show()
