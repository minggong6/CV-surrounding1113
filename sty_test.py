import os

import matplotlib.pyplot as plt # plt 用于显示图片

import matplotlib.image as mpimg # mpimg 用于读取图片

import numpy as np
import imageio
import toolkit_skeleton as tks
import toolkit_3D as tk3

from PIL import Image

root = "annotation_mask"
out = "annotation_mask_out"

img_names = os.listdir(root)
print(img_names)

results = {}

for img_name in img_names:
    img_path = os.path.join(root, img_name)

    img2D = mpimg.imread(img_path) # 读取和代码处于同一目录下的 lena.png

    # img3D = img2D[np.newaxis, :]
    img3D = np.stack((img2D, img2D, img2D))

    skeleton = tks.Skeleton(img3D)

    skeleton.generate_skele_point_list()

    skeleton.generate_skele_point_list()

    skeleton.generate_radius_graph()

    skeleton.generate_path_graph()

    path_graph_3D = skeleton.path_graph

    # path_graph = tk3.image_dilation(path_graph, 3)

    path_graph = np.zeros_like(img2D)

    for x in range(0, path_graph.shape[0]):
        for y in range(0, path_graph.shape[1]):
            z_sum = 0
            for z in range(0, path_graph_3D.shape[0]):
                if path_graph_3D[z, x, y] > 0:
                    z_sum += 1
            if z_sum > 0:
                path_graph[x, y] = 1

    path_graph = path_graph.astype(np.int32)

    px_list = tk3.tuple_to_list(np.where(path_graph > 0))

    surr_list = [(-1, -1), (-1, 1), (1, -1), (1, 1), (0, 1), (0, -1), (1, 0), (-1, 0)]

    for i in range(0, 1):
        for px in px_list:
            surr_sum = 0
            for surr in surr_list:
                new_px = (px[0] + surr[0], px[1] + surr[1])
                surr_sum += path_graph[new_px]
            if surr_sum == 1:
                path_graph[px] = 0

    path_graph_ = path_graph.copy()
    cross_list = []
    for px in px_list:
        surr_sum = 0
        for surr in surr_list:
            new_px = (px[0] + surr[0], px[1] + surr[1])
            if path_graph_[new_px] > 0:
                surr_sum += 1
        if surr_sum > 2:
            cross_list.append(px)
            path_graph_[px] = 0
            for surr in surr_list:
                new_px = (px[0] + surr[0], px[1] + surr[1])
                path_graph_[new_px] = 0

    for px in cross_list:
        for surr in surr_list:
            new_px = (px[0] + surr[0], px[1] + surr[1])
            path_graph[new_px] = 1

    results[img_name] = len(cross_list)

    out_path = os.path.join(out, img_name)
    imageio.imwrite(out_path, path_graph)

print(results)