# 1. 提取全局骨架
# 2. 用血管拆分图筛选出合适的目标血管骨架，并进行再度拆分
# 从已有的拆分图的基础上进行剪枝
import copy
import os.path

import numpy as np
import tqdm
import xlwt
from scipy import ndimage

import toolkit_3D as tk3
import toolkit_main as tkm
import SkeletonAnalysis as skele
from toolkit_skeleton import Skeleton


def get_adjacent_info(skele_graph, curr_coordinate):
    surround_voxel_list = tk3.get_surround_voxel(curr_coordinate, skele_graph.shape)
    adjacent_voxel_list = []
    degree = 0
    for sur_v in surround_voxel_list:
        if skele_graph[sur_v] > 0:
            adjacent_voxel_list.append((sur_v, skele_graph[sur_v]))
            degree += 1
    return degree, adjacent_voxel_list


result_xls_path = "test_data.xls"
book = xlwt.Workbook()
sheet = book.add_sheet('Skeleton Analysis')

artery_part_path = 'data_2_part_artery_remerge-0727'
vein_part_path = 'data_2_part_vein_remerge-0727'
preprocess_path = 'data_p2.4_preprocess'

img_id_list = ['1', '2', '3', '4', '5', '7', '8', '9', '10', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21',
               '24', '25', '26', '27', '29', '30', '31', '32', '33', '35', '36', '38', '39', '40', '41', '42', '43',
               '44', '45', '46', '47', '48', '49', '50', '52', '53', '55', '56', '57', '58', '59', '60', '61', '62',
               '63', '64', '65', '66', '67', '68', '69', '70', '71', '72', '73', '74', '78', '79', '80', '81', '82',
               '83', '84', '86', '103', '104', '107', '108', '110', '111', '112', '114', '116', '117', '118', '121',
               '122', '123']

for img_id in img_id_list:

    artery_part_filename = os.path.join(artery_part_path, img_id + '_artery.nii.gz')
    vein_part_filename = os.path.join(vein_part_path, img_id + '_vein.nii.gz')
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
    img_artery = img_dict['artery']
    img_tumor = img_dict["tumor"]
    img_info = img_dict["info"]
    shape = img_tumor.shape

    img_part = tk3.get_any_nii(artery_part_filename)['img']



    skeleton = Skeleton(img_artery)
    skeleton.generate_skele_point_list()
    skeleton.generate_radius_graph()
    skeleton.generate_path_graph()
    skeleton.generate_cross_block_list()
    skeleton.inject_path_id()
    skeleton.generate_ordered_path()

    copy.deepcopy(skeleton)

    result_img = np.zeros(shape)
    result_img += np.where(img_part == 1, 1, 0)
    # result_img += np.where(img_part == 2, 2, 0)
    result_img += np.where(img_part == 4, 4, 0)
    result_img += np.where(img_part == 5, 5, 0)
    path_img = np.zeros(shape)
    path_radius_img = np.zeros(shape)
    for target_value in [2, 3, 6]:
        img_target = np.where(img_part == target_value, 1, 0)
        img_non_target = np.where(img_part > 0, 1, 0) - img_target
        img_dist = np.multiply(ndimage.distance_transform_edt(np.where(img_non_target > 0, 0, 1)), img_target)
        start_voxel_list = tk3.tuple_to_list(np.where(img_dist == 1))
        start_voxel = start_voxel_list[0]

        part_skeleton = copy.deepcopy(skeleton)
        part_skeleton.vessel = img_target

        part_skeleton.path_graph = np.multiply(part_skeleton.path_graph, img_target)
        part_skeleton.radius_graph = np.multiply(part_skeleton.radius_graph, img_target)

        part_skeleton.trunk_angle_threshold = 180

        part_cross_block_list = []
        for cb in part_skeleton.cross_block_list:
            within_target = False
            for cbv in cb.cross_voxel_list:
                if img_target[cbv] > 0:
                    within_target = True
                    break
            if within_target:
                part_cross_block_list.append(cb)

        part_skeleton.cross_block_list = part_cross_block_list

        part_skeleton.generate_trunk_path_graph(ignore_list=[], contain_list=[], start_vox=start_voxel)
        part_skeleton.generate_part_graph_simulator(part_skeleton.trunk_path_graph)

        target_path_img = np.where(part_skeleton.trunk_path_graph == part_skeleton.part_graph[start_voxel], target_value, 0)
        target_cb_image = np.where(part_skeleton.part_graph == part_skeleton.part_graph[start_voxel], target_value, 0)
        # tk3.save_nii(part_skeleton.part_graph, os.path.join('data_cut_branch_exp', f"{img_id}_artery_cb_{target_value}_{start_voxel}.nii.gz"), img_info)
        # tk3.save_nii(target_cb_image, os.path.join('data_cut_branch_exp', f"{img_id}_artery_cb_{target_value}_{start_voxel}.nii.gz"), img_info)
        result_img += target_cb_image
        path_img += target_path_img
        path_radius_img += np.multiply(np.where(target_path_img > 0, 1, 0), skeleton.radius_graph)
    # tk3.save_nii(tk3.image_dilation(path_img, 2), os.path.join('data_cut_branch_exp', img_id + "_artery_path.nii.gz"), img_info)
    tk3.save_nii(result_img, os.path.join('data_cut_branch_exp', img_id + "_artery_cb.nii.gz"), img_info)
    print('Saving ' + img_id + "_artery_cb.nii.gz")

    # target_path_img = np.where(path_img == 6, 1, 0)
    #
    # illegal_pv_num = 1
    #
    # while illegal_pv_num > 0:
    #
    #     target_pv_list = tk3.tuple_to_list(np.where(target_path_img == 1))
    #
    #     for pv in target_pv_list:
    #         degree, adjacent_voxel_list = get_adjacent_info(target_path_img, pv)
    #         # print(degree, end=" ")
    #         # try to delete voxel degree = 3
    #         remove_pv = False
    #         if degree > 2:
    #             remove_pv = True
    #             target_path_img_trial = target_path_img.copy()
    #             target_path_img_trial[pv] = 0
    #             for pv_adj in adjacent_voxel_list:
    #                 degree_adj, _ = get_adjacent_info(target_path_img_trial, pv_adj[0])
    #                 if degree_adj <= 1:
    #                     remove_pv = False
    #                     break
    #         if remove_pv:
    #             target_path_img[pv] = 0
    #
    #     illegal_pv_num = 0
    #     target_pv_list = tk3.tuple_to_list(np.where(target_path_img == 1))
    #     for pv in target_pv_list:
    #         degree, adjacent_voxel_list = get_adjacent_info(target_path_img, pv)
    #         if degree > 2:
    #             illegal_pv_num += 1
    #
    # # check
    # degree_dict = {}
    # target_pv_list = tk3.tuple_to_list(np.where(target_path_img == 1))
    # end_voxel_list = []
    # for pv in target_pv_list:
    #     degree, adjacent_voxel_list = get_adjacent_info(target_path_img, pv)
    #     if str(degree) in degree_dict.keys():
    #         degree_dict[str(degree)] += 1
    #     else:
    #         degree_dict[str(degree)] = 1
    #     if degree == 1:
    #         end_voxel_list.append(pv)
    # print(f"degree_dict: {degree_dict} {len(target_pv_list)} in total")
    # print(end_voxel_list)
    #
    # # degree visible
    # test_img = np.zeros(shape)
    # target_pv_list = tk3.tuple_to_list(np.where(target_path_img == 1))
    # for pv in target_pv_list:
    #     degree, adjacent_voxel_list = get_adjacent_info(target_path_img, pv)
    #     test_img[pv] = degree
    #     print(degree, end=" ")
    # test_img = tk3.image_dilation(test_img, 2)
    # tk3.save_nii(test_img, os.path.join('data_cut_branch_exp', "test.nii.gz"), img_info)
    #
    # # sequentialize
    # sequence_iter_img = target_path_img.copy()
    # current_pv = end_voxel_list[0]
    # pv_sequence_list = []
    #
    # while True:
    #     degree, adjacent_voxel_list = get_adjacent_info(sequence_iter_img, current_pv)
    #     if degree == 1:
    #         print('successful: ' + str(current_pv))
    #     else:
    #         print('failed: ' + str(current_pv))
    #         pv_sequence_list.append(current_pv)
    #         break
    #     pv_sequence_list.append(current_pv)
    #     sequence_iter_img[current_pv] = 0
    #     current_pv = adjacent_voxel_list[0][0]
    #
    # print(f'pv_sequence_list: {len(pv_sequence_list)} in total')
    #
    # radius_sequence_list = []
    # for pv in pv_sequence_list:
    #     radius = path_radius_img[pv]
    #     radius_sequence_list.append(radius)
    #
    # print(f'radius_sequence_list: {radius_sequence_list}')
    #
    # tumor_dist_graph = ndimage.distance_transform_edt(np.where(img_tumor > 0, 0, 1))
    #
    # dist_sequence_list = []
    # for pv in pv_sequence_list:
    #     dist = tumor_dist_graph[pv]
    #     dist_sequence_list.append(dist)
    #
    # print(f'dist_sequence_list: {dist_sequence_list}')
    #
    # for row in range(0, len(radius_sequence_list)):
    #     sheet.write(row, 0, str(pv_sequence_list[row]))
    #     sheet.write(row, 1, radius_sequence_list[row])
    #     sheet.write(row, 2, dist_sequence_list[row])
    #
    # book.save(result_xls_path)
