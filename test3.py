import os

import numpy as np
from scipy.spatial.transform import Rotation
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

matrix = np.array([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9]])

# 指定旋转轴的向量
axis_vector = np.array([0.5, 0.5, 0.5])

# 指定旋转的角度（以度为单位）
angle_degrees = 30.0

# 将角度从度转换为弧度
angle_radians = np.radians(angle_degrees)

# 创建 Rotation 对象，表示绕指定轴的旋转
rotation = Rotation.from_rotvec(angle_radians * axis_vector)

# 使用 apply 方法将矩阵应用于旋转
rotated_matrix = rotation.apply(img)

print("原始矩阵：")
print(matrix)
print("旋转后的矩阵：")
print(rotated_matrix)
tk3.save_nii(rotated_matrix, "test1.nii.gz", img_info)