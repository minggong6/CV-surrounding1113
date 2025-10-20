import cv2
from skimage import morphology
import numpy as np


img = cv2.imread('test_img.png', 0)  # 读取图片
_, binary = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY_INV)  # 二值化处理
cv2.imwrite("binary.png", binary)  # 保存二值化图片

binary[binary == 0] = 1
binary[binary == 255] = 0
skeleton0 = morphology.skeletonize(binary)  # 骨架提取
skeleton = skeleton0.astype(np.uint8) * 255
cv2.imwrite("skeleton.png", skeleton)  # 保存骨架提取后的图片
