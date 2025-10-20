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


def get_img_file_list(data_list_path):
    img_name_list = read_nii_list(data_list_path)

    file_list = []

    for img_path in img_name_list:
        img_name = os.path.basename(img_path)
        img_id = img_name.split(sep="_")[0]
        img_id = img_id.split(sep=".")[0]
        file_dict = {
            "img_id": img_id,
            "img_path": img_path,
            "img_contact_path": None
        }
        file_list.append(file_dict)
    return file_list


def get_img_file_list_old(dataset_path):
    file_list = []
    img_name_list = os.listdir(dataset_path)
    for img_name in img_name_list:
        img_id = img_name.split(sep="_")[0]

        # for original newdata
        # img_id = img_name.split(sep="_")[1]
        # img_id = img_id.split(sep=".")[0]
        img_path = os.path.join(dataset_path, img_name)
        file_dict = {
            "img_id": img_id,
            "img_path": img_path,
            "img_contact_path": None
        }
        file_list.append(file_dict)
    return file_list


def get_brief(brief_path):
    brief_book = xlrd2.open_workbook(brief_path)
    brief_sheet = brief_book.sheet_by_index(0)

    case_list = []
    for row in range(1, brief_sheet.nrows):
        case_id = int(brief_sheet.cell_value(row, 0))
        target = str(brief_sheet.cell_value(row, 1))
        wrapping = str(brief_sheet.cell_value(row, 2))
        deformation = str(brief_sheet.cell_value(row, 3))
        case_dict = {
            "case_id": case_id,
            "target": target,
            "wrapping": wrapping,
            "deformation": deformation
        }
        case_list.append(case_dict)
    return case_list


def get_thin_GT(thin_GT_path):
    thin_GT_book = xlrd2.open_workbook(thin_GT_path)
    thin_GT_sheet = thin_GT_book.sheet_by_index(0)

    GT_dict = {}
    for row in range(1, thin_GT_sheet.nrows):
        raw_case_id = thin_GT_sheet.cell_value(row, 0)
        if isinstance(raw_case_id, float) or isinstance(raw_case_id, int):
            case_id = str(int(raw_case_id))
        else:
            case_id = raw_case_id
        artery_GT = int(thin_GT_sheet.cell_value(row, 1))
        vein_GT = int(thin_GT_sheet.cell_value(row, 2))
        GT_dict[case_id] = (artery_GT, vein_GT)
    return GT_dict


def get_brief_info(brief_list, case_id, target):
    for brief_dict in brief_list:
        if brief_dict["case_id"] == int(case_id):
            if brief_dict["target"] == str(target):
                return brief_dict["wrapping"], brief_dict["deformation"]


def generate_contour_nii_3D(file_dict, new_dir_path, prefix="contact_", contour_thickness=1.5, contact_range=2,
                            axis='z'):
    """
    Save contour image as [prefix + new_filename]
    Update ["img_contact_path"] in [file_dict]

        background - 0
        tumor - 1
        contour_vein - 2
        contact_vein - 3
        contour_artery - 4
        contact_artery - 5

    :param axis:
    :param file_dict:
    :param new_dir_path:
    :param prefix:
    :param contour_thickness:
    :param contact_range:
    :return:
    """

    # if [new_filename] is given, then save as [new_filename]
    # else save as the original path: "xxx/xxx/prefix + xxx.nii.gz"
    img_full_path = file_dict["img_path"]
    img_dir_path = os.path.split(img_full_path)[0]
    img_filename = str(os.path.split(img_full_path)[1])
    new_filename = prefix + img_filename
    new_full_path = os.path.join(new_dir_path, new_filename)

    # read nii image
    img_dict = tk3.get_nii(img_full_path, axis=axis)

    # get target contour
    contour_vein = tk3.get_2D_contour(img_dict["vein"], contour_thickness=contour_thickness)
    contour_artery = tk3.get_2D_contour(img_dict["artery"], contour_thickness=contour_thickness)

    # get target-tumor contact
    tumor = img_dict["tumor"]
    contact_vein = tk3.get_3D_contact(tumor, contour_vein, contact_range=contact_range)
    contact_artery = tk3.get_3D_contact(tumor, contour_artery, contact_range=contact_range)

    new_data = tumor * 1 + contour_vein * 2 + contact_vein + contour_artery * 4 + contact_artery

    file_dict["img_contact_path"] = new_full_path

    tk3.save_nii(new_data, new_full_path, img_dict["info"], axis=axis)


def generate_contour_nii_2D(file_dict, new_dir_path, prefix="contact_", contour_thickness=1.5, contact_range=2):
    """
    Save contour image as [prefix + new_filename]
    Update ["img_contact_path"] in [file_dict]

        background - 0
        tumor - 1
        contour_vein - 2
        contact_vein - 3
        contour_artery - 4
        contact_artery - 5

    :param file_dict:
    :param new_dir_path:
    :param prefix:
    :param contour_thickness:
    :param contact_range:
    :return:
    """

    # if [new_filename] is given, then save as [new_filename]
    # else save as the original path: "xxx/xxx/prefix + xxx.nii.gz"
    img_full_path = file_dict["img_path"]
    img_dir_path = os.path.split(img_full_path)[0]
    img_filename = str(os.path.split(img_full_path)[1])
    new_filename = prefix + img_filename
    new_full_path = os.path.join(new_dir_path, new_filename)
    strategy = '2D'

    # read nii image
    img_dict = tk3.get_nii(img_full_path)

    # get target contour
    if strategy == '3D':
        # choice 1: use 3D overview contour
        contour_vein = tk3.get_3D_contour(img_dict["vein"], contour_thickness=contour_thickness)
        contour_artery = tk3.get_3D_contour(img_dict["artery"], contour_thickness=contour_thickness)
    else:
        # choice 2: use 2D per slice contour
        contour_vein = tk3.get_3D_contour(img_dict["vein"], contour_thickness=contour_thickness)
        contour_artery = tk3.get_3D_contour(img_dict["artery"], contour_thickness=contour_thickness)

    # get target-tumor contact
    if strategy == '3D':
        tumor = img_dict["tumor"]
        contact_vein = tk3.get_3D_contact(tumor, contour_vein, contact_range=contact_range)
        contact_artery = tk3.get_3D_contact(tumor, contour_artery, contact_range=contact_range)
    else:
        tumor = img_dict["tumor"]
        contact_vein = tk3.get_3D_contact(tumor, contour_vein, contact_range=contact_range)
        contact_artery = tk3.get_3D_contact(tumor, contour_artery, contact_range=contact_range)

    new_data = tumor * 1 + contour_vein * 2 + contact_vein + contour_artery * 4 + contact_artery

    file_dict["img_contact_path"] = new_full_path

    tk3.save_nii(new_data, new_full_path, img_dict["info"])


def calculate_2D_contact(file_dict, target, contact_dir=None, size_threshold=0, axis='z'):
    def get_sep_list(img, size_threshold=0):
        def cut_piece(img, x, y):
            sep = np.zeros(img.shape)
            iter(img, sep, x, y)
            return sep

        def iter(img, res, x, y):
            if img[x, y] == 0:
                return
            else:
                img[x, y] = 0
                res[x, y] = 1
                if x > 0:
                    iter(img, res, x - 1, y)
                if y > 0:
                    iter(img, res, x, y - 1)
                if x < height - 1:
                    iter(img, res, x + 1, y)
                if y < width - 1:
                    iter(img, res, x, y + 1)
                if x < height - 1 and y < width - 1:
                    iter(img, res, x + 1, y + 1)
                if x > 0 and y < width - 1:
                    iter(img, res, x - 1, y + 1)
                if x < height - 1 and y > 0:
                    iter(img, res, x + 1, y - 1)
                if x > 0 and y > 0:
                    iter(img, res, x - 1, y - 1)
                return

        def get_piece_size(piece):
            return np.sum(piece == 1)

        matrix = img.copy()
        height = matrix.shape[0]
        width = matrix.shape[1]
        sep_list = []
        while True:
            if np.sum(matrix == 1) <= 0:
                break
            else:
                indices = np.where(matrix == 1)
                piece = cut_piece(matrix, indices[0][0], indices[1][0])
                piece_size = get_piece_size(piece)

                if piece_size <= size_threshold:
                    continue
                else:
                    sep_list.append(piece)

        return sep_list

    min_contour_px_num = 5
    min_ratio = 0.1

    contact_img_path = file_dict["img_contact_path"]
    if contact_img_path is None:
        contact_img_path = os.path.join(contact_dir, 'contact_' + os.path.basename(file_dict["img_path"]))
    img_contact_dict = tk3.get_contour_nii(contact_img_path, axis=axis)
    """
        background - 0
        tumor - 1
        contour_vein - 2
        contact_vein - 3
        contour_artery - 4
        contact_artery - 5
    """
    img_contour = img_contact_dict["contour_" + target]
    img_contact = img_contact_dict["contact_" + target]

    total_slice = img_contour.shape[0]

    max_ratio = 0
    max_slice = -1

    # save result of every slice in one case
    slice_result_list = []
    for slice_num in range(0, total_slice):
        # For every slice
        img_contour_slice = img_contour[slice_num, :, :]
        img_contact_slice = img_contact[slice_num, :, :]
        sep_target_list = get_sep_list(img_contour_slice, size_threshold)

        slice_max_ratio = 0
        max_ratio_contour = 0
        max_ratio_contact = 0

        # save result of every separate part in one slice
        part_result_list = []
        for img_sep_contour_slice in sep_target_list:
            # For every part
            contour_px_num = np.sum(img_sep_contour_slice[img_sep_contour_slice == 1])
            if contour_px_num <= min_contour_px_num:
                continue

            # calculate roundness
            vessel_slice = tk3.fill_hole(img_sep_contour_slice)
            area = np.sum(vessel_slice[vessel_slice == 1])
            circumference = contour_px_num
            roundness = circumference ** 2 / (area * 12.56)

            img_sep_contact_slice = np.multiply(img_sep_contour_slice, img_contact_slice)
            contact_px_num = np.sum(img_sep_contact_slice[img_sep_contact_slice == 1])

            if contact_px_num == 0 or contour_px_num == 0:
                ratio = 0
            else:
                ratio = contact_px_num / contour_px_num

            if ratio <= min_ratio:
                continue

            if ratio > max_ratio:
                max_ratio = ratio
                max_slice = slice_num

            if ratio > slice_max_ratio:
                slice_max_ratio = ratio

            slice_result_dict = {
                "slice_num": slice_num,
                "slice_max_ratio": slice_max_ratio,
                "part_contour": contour_px_num,
                "part_contact": contact_px_num,
                "part_ratio": ratio,
                "part_area": area,
                "part_roundness": roundness
            }
            slice_result_list.append(slice_result_dict)

    # if every slice is empty slice, mark the case with slice_num = -1
    if len(slice_result_list) == 0:
        slice_result_dict = {
            "slice_num": -1,
            "slice_max_ratio": 0,
            "part_contour": 0,
            "part_contact": 0,
            "part_ratio": 0,
            "part_area": 0,
            "part_roundness": 0
        }
        slice_result_list.append(slice_result_dict)
    return slice_result_list, max_ratio, max_slice


def calculate_2D_contact_1(file_dict, target, contact_dir=None, size_threshold=0, axis='z'):
    """
    Edition 2, updated on 2023.6.7, for demo_surrounding.py
    :param file_dict:
    :param target:
    :param contact_dir:
    :param size_threshold:
    :param axis:
    :return:
    """
    def get_sep_list(img, size_threshold=0):
        def cut_piece(img, x, y):
            sep = np.zeros(img.shape)
            iter(img, sep, x, y)
            return sep

        def iter(img, res, x, y):
            if img[x, y] == 0:
                return
            else:
                img[x, y] = 0
                res[x, y] = 1
                if x > 0:
                    iter(img, res, x - 1, y)
                if y > 0:
                    iter(img, res, x, y - 1)
                if x < height - 1:
                    iter(img, res, x + 1, y)
                if y < width - 1:
                    iter(img, res, x, y + 1)
                if x < height - 1 and y < width - 1:
                    iter(img, res, x + 1, y + 1)
                if x > 0 and y < width - 1:
                    iter(img, res, x - 1, y + 1)
                if x < height - 1 and y > 0:
                    iter(img, res, x + 1, y - 1)
                if x > 0 and y > 0:
                    iter(img, res, x - 1, y - 1)
                return

        def get_piece_size(piece):
            return np.sum(piece == 1)

        matrix = img.copy()
        height = matrix.shape[0]
        width = matrix.shape[1]
        sep_list = []
        while True:
            if np.sum(matrix == 1) <= 0:
                break
            else:
                indices = np.where(matrix == 1)
                piece = cut_piece(matrix, indices[0][0], indices[1][0])
                piece_size = get_piece_size(piece)

                if piece_size <= size_threshold:
                    continue
                else:
                    sep_list.append(piece)

        return sep_list

    min_contour_px_num = 5
    min_ratio = 0.1

    contact_img_path = file_dict["img_contact_path"]
    if contact_img_path is None:
        contact_img_path = os.path.join(contact_dir, 'contact_' + os.path.basename(file_dict["img_path"]))
    img_contact_dict = tk3.get_contour_nii(contact_img_path, axis=axis)
    """
        background - 0
        tumor - 1
        contour_vein - 2
        contact_vein - 3
        contour_artery - 4
        contact_artery - 5
    """
    img_contour = img_contact_dict["contour_" + target]
    img_contact = img_contact_dict["contact_" + target]

    total_slice_num = img_contour.shape[0]

    max_ratio = 0
    max_slice = -1

    # save result of every slice in one case
    slice_result_list = []
    for slice_id in range(0, total_slice_num):
        # For every slice
        img_contour_slice = img_contour[slice_id, :, :]
        img_contact_slice = img_contact[slice_id, :, :]
        print(file_dict['img_id'] + ' slice size' + str(img_contact_slice.shape))
        sep_target_list = get_sep_list(img_contour_slice, size_threshold)

        slice_max_ratio = 0

        # save result of every separate part in one slice
        for img_sep_contour_slice in sep_target_list:
            # For every part
            contour_px_num = np.sum(img_sep_contour_slice[img_sep_contour_slice == 1])
            if contour_px_num <= min_contour_px_num:
                continue

            # print(np.where(img_sep_contour_slice == 1))

            px_list = tk3.tuple_to_list(np.where(img_sep_contour_slice == 1))

            position = (slice_id, px_list[0][0], px_list[0][1])
            # for px in px_list:
            #     position = (position[0] + px[0], position[1] + px[1])
            # position = (slice_id,
            #             int(position[0] / len(px_list)),
            #             int(position[1] / len(px_list)))

            # calculate roundness
            vessel_slice = tk3.fill_hole(img_sep_contour_slice)
            area = np.sum(vessel_slice[vessel_slice == 1])
            circumference = contour_px_num
            roundness = circumference ** 2 / (area * 12.56)

            img_sep_contact_slice = np.multiply(img_sep_contour_slice, img_contact_slice)
            contact_px_num = np.sum(img_sep_contact_slice[img_sep_contact_slice == 1])

            if contact_px_num == 0 or contour_px_num == 0:
                ratio = 0
            else:
                ratio = contact_px_num / contour_px_num

            if ratio <= min_ratio:
                continue

            if ratio > max_ratio:
                max_ratio = ratio
                max_slice = slice_id

            if ratio > slice_max_ratio:
                slice_max_ratio = ratio

            slice_result_dict = {
                "slice_num": slice_id,
                "slice_max_ratio": slice_max_ratio,
                "part_contour": contour_px_num,
                "part_contact": contact_px_num,
                "part_ratio": ratio,
                "part_area": area,
                "part_roundness": roundness,
                "part_position": position
            }
            slice_result_list.append(slice_result_dict)

    # if every slice is empty slice, mark the case with slice_num = -1
    if len(slice_result_list) == 0:
        slice_result_dict = {
            "slice_num": -1,
            "slice_max_ratio": 0,
            "part_contour": 0,
            "part_contact": 0,
            "part_ratio": 0,
            "part_area": 0,
            "part_roundness": 0,
            "part_position": (-1, -1, -1)
        }
        slice_result_list.append(slice_result_dict)
    return slice_result_list, max_ratio, max_slice


def thin_detect(part_img, path_list, img_tumor,
                thin_threshold=2,
                tumor_distance_m1=5,
                tumor_distance_m2=2,
                isl_max_num=3,
                img_info=None,
                target="vein",
                method="both"):
    """

    :param tumor_distance:
    :param part_img: the output of [skele.distance_to_category]
    :param path_list:
    :param thin_threshold:
    :param isl_max_num: the maximum of island number, such as (2, 3), None for no restrain
    :return:
    """

    def thin_locate_m1(part_erode, path_point_list):
        path_start = 0
        path_end = len(path_point_list)
        path_range = list(range(0, len(path_point_list)))
        thin_point_list = []

        # Clear the path point in the end of the part's path
        for i in path_range:
            if part_erode[path_point_list[i]] == 0:
                path_start += 1
            else:
                break
        for i in path_range[::-1]:
            if part_erode[path_point_list[i]] == 0:
                path_end -= 1
            else:
                break

        # Find the path point in the eroded break position
        for i in range(path_start, path_end):
            path_point = path_point_list[i]

            if part_eroded[path_point] == 1:
                thin_point_list.append(path_point)

        if len(thin_point_list) == 0:
            return None
        else:
            return thin_point_list

    def thin_locate_m2(part_erode, part_eroded, max_dist_minus=0.8):
        n, islands = tk3.get_islands_num(part_erode)
        dist_map_list = []
        minus_map_total = np.zeros(part_erode.shape)
        for i in range(1, n + 1):
            dist_map = np.where(islands == i, 0, 1)
            dist_map = ndimage.distance_transform_edt(dist_map)
            dist_map = np.multiply(dist_map, part_eroded)
            dist_map_list.append(dist_map)
            if i == 1:
                continue
            for j in range(0, i - 1):
                minus_map = np.abs(dist_map_list[j] - dist_map)
                minus_map = np.where(minus_map < max_dist_minus, 1, 0)
                minus_map = np.multiply(minus_map, part_eroded)
                minus_map_total += minus_map
        if np.sum(minus_map_total[minus_map_total > 0]) > 0:
            result = np.where(minus_map_total > 0, 0, 1)
            return result
        else:
            return None

    def get_max_radis(part):
        radis = 0
        while True:
            part, _ = tk3.erode(part, times=1)
            radis += 1
            if np.sum(part[part > 0]) == 0:
                break
        return radis

    part_id_range = range(1, int(np.max(part_img)) + 1)
    result_img = np.zeros(part_img.shape)
    result_info_list = []

    for idx, part_id in enumerate(tqdm.tqdm(part_id_range, desc="thin detect")):
        # for part_id in part_id_range:
        # For every vessel part in the [part_img]
        part = np.where(part_img == part_id, 1, 0)
        thin_degree = 0
        thin_locate_result_m1 = None
        thin_locate_result_m2 = None

        # tumor_distance = path_list[part_id - 1]["weight"] + 3

        # tk3.save_nii(part, "suspect thin/part_" + str(part_id) + "_erode_" + str(0) + ".nii.gz", img_info)
        for thin_times in range(1, thin_threshold + 1):
            # Erode the part from outside layer by layer
            part_erode, part_eroded = tk3.erode(part, thin_times)
            # tk3.save_nii(part_erode, "suspect thin/part_" + str(part_id) + "_erode_" + str(thin_times) + ".nii.gz", img_info)

            # Exclude too small vessels and extreme condition
            # Restrain the broking into islands number
            isl_num, isl_img = tk3.get_islands_num(part_erode)
            # print("    " + str(part_id) + " - isl_num = " + str(isl_num))
            if isl_num == 0:
                break
            if isl_num == 1:
                continue
            if isl_max_num is not None:
                if isl_num > isl_max_num:
                    break
            # Once the one connect part is eroded into several parts
            thin_degree = thin_times

            thin_locate_result_m1 = thin_locate_m1(part_erode, path_list[part_id - 1]["path"])
            thin_locate_result_m2 = thin_locate_m2(part_erode, part_eroded)

            if thin_locate_result_m1 is None and thin_locate_result_m2 is None:
                continue

            result_1 = False
            result_2 = False

            do_m1 = True
            do_m2 = True
            if method == "both":
                pass
            elif method == "m1":
                do_m1 = True
                do_m2 = False
            elif method == "m2":
                do_m1 = False
                do_m2 = True

            # Judging method 2
            if not do_m2:
                pass
            elif thin_locate_result_m2 is not None:
                distance_map = ndimage.distance_transform_edt(thin_locate_result_m2)
                distance_map = np.multiply(distance_map, img_tumor)
                distance_map = np.where(distance_map == 0, tumor_distance_m2 + 1, distance_map)

                if np.sum(distance_map[distance_map <= tumor_distance_m2]) > 0:
                    result_2 = True
                    # result_img += np.where(part_img == part_id, part_img, 0)
                    if target == "vein":
                        result_img += np.where(part_img == part_id, 3, 0)
                        result_img += np.where(thin_locate_result_m2 == 0, 2, 0)
                    elif target == "artery":
                        result_img += np.where(part_img == part_id, 1, 0)
                        result_img += np.where(thin_locate_result_m2 == 0, 3, 0)

            # Judging method 1
            if not do_m1:
                pass
            elif thin_locate_result_m1 is not None and not result_2:
                point_list_map = np.ones(part_img.shape)
                for thin_point in thin_locate_result_m1:
                    point_list_map[thin_point] = 0
                distance_map = ndimage.distance_transform_edt(point_list_map)
                distance_map = np.multiply(distance_map, img_tumor)
                distance_map = np.where(distance_map == 0, tumor_distance_m1 + 1, distance_map)

                if np.sum(distance_map[distance_map <= tumor_distance_m1]) > 0:
                    result_1 = True
                    # result_img += np.where(part_img == part_id, part_img, 0)
                    if target == "vein":
                        result_img += np.where(part_img == part_id, 3, 0)
                        result_img += np.where(point_list_map == 0, 4, 0)
                    elif target == "artery":
                        result_img += np.where(part_img == part_id, 1, 0)
                        result_img += np.where(point_list_map == 0, 5, 0)

            if result_1 or result_2:
                result_info_dict = {
                    "part_id": part_id,
                    "thin_degree": thin_degree,
                    # "part_radis": path_list[part_id - 1]["weight"]
                    "part_radis": get_max_radis(part)
                }
                result_info_list.append(result_info_dict)

            break

    if np.sum(result_img[result_img > 0]) > 0:
        return result_img, result_info_list
    else:
        return None, None


def slice_filter(img0, threshold_size_tuple=(10, 5, 5), suspect_size=3000, compensate_dist=5, keep_list=None):
    img = img0.copy()
    shape = img.shape

    keep_mask = np.zeros(shape)
    if keep_list is not None:
        for (point, radis) in keep_list:
            keep_mask += tk3.get_sphere_mask(point, shape, radis)
        keep_mask = np.where(keep_mask > 0, 1, 0)
        keep_mask = np.multiply(keep_mask, img)

    for slice_num in range(0, shape[0]):
        slice = img[slice_num, :, :]
        slice = tk3.remove_islands(slice, threshold_size=threshold_size_tuple[0])
        img[slice_num, :, :] = slice
    img = img + keep_mask
    img = np.where(img > 0, 1, 0)
    img = tk3.remove_islands(img, threshold_size=suspect_size)

    for slice_num in range(0, shape[1]):
        slice = img[:, slice_num, :]
        slice = tk3.remove_islands(slice, threshold_size=threshold_size_tuple[1])
        img[:, slice_num, :] = slice
    img = img + keep_mask
    img = np.where(img > 0, 1, 0)
    img = tk3.remove_islands(img, threshold_size=suspect_size)

    for slice_num in range(0, shape[2]):
        slice = img[:, :, slice_num]
        slice = tk3.remove_islands(slice, threshold_size=threshold_size_tuple[2])
        img[:, :, slice_num] = slice
    img = img + keep_mask
    img = np.where(img > 0, 1, 0)
    img = tk3.remove_islands(img, threshold_size=suspect_size)

    if compensate_dist > 0:
        dist_map = ndimage.distance_transform_edt(np.where(img == 1, 0, 1))
        dist_map = np.multiply(dist_map, img0)
        dist_map = np.where(dist_map == 0, compensate_dist + 1, dist_map)
        compensate_map = np.where(dist_map <= compensate_dist, 1, 0)
        img += compensate_map

    img = tk3.remove_islands(img, threshold_size=suspect_size)

    return img


def get_keep_list(img_id, target):
    artery_dict = {
        "15": [((54, 107, 118), 5), ((52, 109, 106), 5)],
        "82b": [((82, 69, 184), 10)],
        "84": [((87, 81, 195), 10)]
    }

    vein_dict = {

    }

    if target == "artery":
        try:
            keep_list = artery_dict[img_id]
        except KeyError:
            keep_list = None
    elif target == "vein":
        try:
            keep_list = vein_dict[img_id]
        except KeyError:
            keep_list = None
    else:
        keep_list = None

    return keep_list


def get_suspect_size(img_id, target, default_suspect_size=3000):
    artery_dict = {

    }

    vein_dict = {
        "12": 1000,
        "14": 1000,
        "27": 1000,
        "32": 1000,
        "35": 1500,
    }

    if target == "artery":
        try:
            suspect_size = artery_dict[img_id]
        except KeyError:
            suspect_size = default_suspect_size
    elif target == "vein":
        try:
            suspect_size = vein_dict[img_id]
        except KeyError:
            suspect_size = default_suspect_size
    else:
        suspect_size = default_suspect_size

    return suspect_size


def get_tumor_info(tumor, spacing=(1.0, 1.0, 1.0)):
    surface = np.where(tumor > 0, 0, 1)

    surface = np.where(ndimage.distance_transform_edt(surface) == 1, 1, 0)

    max_diameter = 0
    diameter_voxel_pair = None

    surface_voxel_list = tk3.tuple_to_list(np.where(surface > 0))

    for i in range(0, len(surface_voxel_list)):
        if i + 1 < len(surface_voxel_list):
            for j in range(i + 1, len(surface_voxel_list)):
                diameter = tk3.get_distance(surface_voxel_list[i], surface_voxel_list[j], spacing)
                if diameter > max_diameter:
                    max_diameter = diameter
                    diameter_voxel_pair = (surface_voxel_list[i], surface_voxel_list[j])
                    print('Coordinate: ' + str(surface_voxel_list[i]) + '-' + str(
                        surface_voxel_list[j]) + '  Diameter: ' + str(diameter))

    volume = np.sum(tumor > 0) * spacing[0] * spacing[1] * spacing[2]

    return {'diameter': max_diameter,
            'diameter_voxels': diameter_voxel_pair,
            'volume': volume
            }


class Contain:
    def __init__(self):
        self.target_dict = {
            'AO': 1,  # 腹主动脉
            'CA': 2,  # 腹腔干
            'LGA': 3,  # 胃左动脉
            'SA': 4,  # 脾动脉
            'RHA': 5,  # 肝右动脉
            'SMA': 6,  # 肠系膜上动脉
            'other1': 7,
            'other2': 8,
            'other3': 9,
            'other4': 10,
        }

        self.contain_dict = {
            '0': {'AO': [],  # 腹主动脉
                  'CA': [],  # 腹腔干
                  'LGA': [],  # 胃左动脉
                  'SA': [],  # 脾动脉
                  'RHA': [],  # 肝右动脉
                  'SMA': [],  # 肠系膜上动脉
                  'other1': [],
                  'other2': [],
                  'other3': [],
                  'other4': [],
                  },
            '1': {'AO': [34, 35, 1],  # 腹主动脉
                  'CA': [32],  # 腹腔干
                  'LGA': [16],  # 胃左动脉
                  'SA': [19],  # 脾动脉
                  'RHA': [31],  # 肝右动脉
                  'SMA': [39],  # 肠系膜上动脉
                  'other1': [],
                  'other2': [],
                  'other3': [],
                  'other4': [],
                  },
            '2': {'AO': [1],  # 腹主动脉
                  'CA': [3, 10],  # 腹腔干
                  'LGA': [4],  # 胃左动脉
                  'SA': [9],  # 脾动脉
                  'RHA': [],  # 肝右动脉
                  'SMA': [2],  # 肠系膜上动脉
                  'other1': [],
                  'other2': [],
                  'other3': [],
                  'other4': [],
                  },
            '3': {'AO': [13, 14],  # 腹主动脉
                  'CA': [11],  # 腹腔干
                  'LGA': [5],  # 胃左动脉
                  'SA': [6],  # 脾动脉
                  'RHA': [10],  # 肝右动脉
                  'SMA': [31],  # 肠系膜上动脉
                  'other1': [],
                  'other2': [],
                  'other3': [],
                  'other4': [],
                  },
            '4': {'AO': [15, 17],  # 腹主动脉
                  'CA': [14],  # 腹腔干
                  'LGA': [10],  # 胃左动脉
                  'SA': [13],  # 脾动脉
                  'RHA': [],  # 肝右动脉
                  'SMA': [28],  # 肠系膜上动脉
                  'other1': [26],
                  'other2': [8],
                  'other3': [],
                  'other4': [],
                  },
            '5': {'AO': [1],  # 腹主动脉
                  'CA': [11],  # 腹腔干
                  'LGA': [3],  # 胃左动脉
                  'SA': [5],  # 脾动脉
                  'RHA': [19],  # 肝右动脉
                  'SMA': [2, 4],  # 肠系膜上动脉
                  'other1': [6],
                  'other2': [7],
                  'other3': [],
                  'other4': [],
                  },
            '7': {'AO': [37, 38],  # 腹主动脉
                  'CA': [36],  # 腹腔干
                  'LGA': [29],  # 胃左动脉
                  'SA': [34],  # 脾动脉
                  'RHA': [24, 35],  # 肝右动脉
                  'SMA': [12],  # 肠系膜上动脉
                  'other1': [39],
                  'other2': [23],
                  'other3': [],
                  'other4': [],
                  },
            '8': {'AO': [17, 1],  # 腹主动脉
                  'CA': [16],  # 腹腔干
                  'LGA': [7],  # 胃左动脉
                  'SA': [14],  # 脾动脉
                  'RHA': [15],  # 肝右动脉
                  'SMA': [25],  # 肠系膜上动脉
                  'other1': [18],
                  'other2': [4],
                  'other3': [],
                  'other4': [],
                  },
            '9': {'AO': [49],  # 腹主动脉
                  'CA': [43],  # 腹腔干
                  'LGA': [31],  # 胃左动脉
                  'SA': [42],  # 脾动脉
                  'RHA': [],  # 肝右动脉
                  'SMA': [19],  # 肠系膜上动脉
                  'other1': [48],
                  'other2': [4],
                  'other3': [],
                  'other4': [],
                  },
            '10': {'AO': [15],  # 腹主动脉
                   'CA': [14],  # 腹腔干
                   'LGA': [13],  # 胃左动脉
                   'SA': [10],  # 脾动脉
                   'RHA': [],  # 肝右动脉
                   'SMA': [21],  # 肠系膜上动脉
                   'other1': [16],
                   'other2': [9],
                   'other3': [],
                   'other4': [],
                   },
            '12': {'AO': [11],  # 腹主动脉
                   'CA': [34, 32, 31],  # 腹腔干
                   'LGA': [23],  # 胃左动脉
                   'SA': [30],  # 脾动脉
                   'RHA': [],  # 肝右动脉
                   'SMA': [9],  # 肠系膜上动脉
                   'other1': [10],
                   'other2': [6],
                   'other3': [8],
                   'other4': [],
                   },
            '13': {'AO': [7],  # 腹主动脉
                   'CA': [16, 18],  # 腹腔干
                   'LGA': [12],  # 胃左动脉
                   'SA': [15],  # 脾动脉
                   'RHA': [17],  # 肝右动脉
                   'SMA': [19],  # 肠系膜上动脉
                   'other1': [3],
                   'other2': [6],
                   'other3': [],
                   'other4': [],
                   },
            '14': {'AO': [32, 58, 59],  # 腹主动脉
                   'CA': [55, 56],  # 腹腔干
                   'LGA': [52, 54],  # 胃左动脉
                   'SA': [49],  # 脾动脉
                   'RHA': [33],  # 肝右动脉
                   'SMA': [26],  # 肠系膜上动脉
                   'other1': [31],
                   'other2': [9],
                   'other3': [],
                   'other4': [],
                   },
            '15': {'AO': [33, 31],  # 腹主动脉
                   'CA': [29],  # 腹腔干
                   'LGA': [30],  # 胃左动脉
                   'SA': [52],  # 脾动脉
                   'RHA': [],  # 肝右动脉
                   'SMA': [28],  # 肠系膜上动脉
                   'other1': [43],
                   'other2': [7],
                   'other3': [],
                   'other4': [],
                   },
            '16': {'AO': [19, 18, 14],  # 腹主动脉
                   'CA': [17, 25, 24],  # 腹腔干
                   'LGA': [46],  # 胃左动脉
                   'SA': [36],  # 脾动脉
                   'RHA': [],  # 肝右动脉
                   'SMA': [20],  # 肠系膜上动脉
                   'other1': [12],
                   'other2': [23],
                   'other3': [],
                   'other4': [],
                   },
            '17': {'AO': [21, 22],  # 腹主动脉
                   'CA': [20, 1, 19],  # 腹腔干
                   'LGA': [18],  # 胃左动脉
                   'SA': [12],  # 脾动脉
                   'RHA': [],  # 肝右动脉
                   'SMA': [33],  # 肠系膜上动脉
                   'other1': [35],
                   'other2': [34],
                   'other3': [],
                   'other4': [],
                   },
            '18': {'AO': [4],  # 腹主动脉
                   'CA': [13],  # 腹腔干
                   'LGA': [5],  # 胃左动脉
                   'SA': [12],  # 脾动脉
                   'RHA': [],  # 肝右动脉
                   'SMA': [17],  # 肠系膜上动脉
                   'other1': [3],
                   'other2': [14],
                   'other3': [],
                   'other4': [],
                   },
            '19': {'AO': [55, 51, 53],  # 腹主动脉
                   'CA': [50, 46, 44, 45],  # 腹腔干
                   'LGA': [43, 25],  # 胃左动脉
                   'SA': [42],  # 脾动脉
                   'RHA': [6],  # 肝右动脉
                   'SMA': [49],  # 肠系膜上动脉
                   'other1': [5],
                   'other2': [54],
                   'other3': [],
                   'other4': [],
                   },
            '20': {'AO': [6, 5, 3],  # 腹主动脉
                   'CA': [39, 40, 41],  # 腹腔干
                   'LGA': [28],  # 胃左动脉
                   'SA': [38],  # 脾动脉
                   'RHA': [],  # 肝右动脉
                   'SMA': [15],  # 肠系膜上动脉
                   'other1': [43],
                   'other2': [48],
                   'other3': [],
                   'other4': [],
                   },
            '21': {'AO': [15],  # 腹主动脉
                   'CA': [22],  # 腹腔干
                   'LGA': [16],  # 胃左动脉
                   'SA': [39],  # 脾动脉
                   'RHA': [21],  # 肝右动脉
                   'SMA': [44],  # 肠系膜上动脉
                   'other1': [46],
                   'other2': [8],
                   'other3': [],
                   'other4': [],
                   },
            '24': {'AO': [13],  # 腹主动脉
                   'CA': [48],  # 腹腔干
                   'LGA': [16],  # 胃左动脉
                   'SA': [39],  # 脾动脉
                   'RHA': [],  # 肝右动脉
                   'SMA': [21, 22],  # 肠系膜上动脉
                   'other1': [44],
                   'other2': [46],
                   'other3': [],
                   'other4': [],
                   },
            '25': {'AO': [26],  # 腹主动脉
                   'CA': [12, 24],  # 腹腔干
                   'LGA': [13],  # 胃左动脉
                   'SA': [23],  # 脾动脉
                   'RHA': [],  # 肝右动脉
                   'SMA': [11],  # 肠系膜上动脉
                   'other1': [8],
                   'other2': [25],
                   'other3': [],
                   'other4': [],
                   },
            '26': {'AO': [4],  # 腹主动脉
                   'CA': [40, 39],  # 腹腔干
                   'LGA': [38],  # 胃左动脉
                   'SA': [33],  # 脾动脉
                   'RHA': [],  # 肝右动脉
                   'SMA': [26],  # 肠系膜上动脉
                   'other1': [13],
                   'other2': [1],
                   'other3': [],
                   'other4': [],
                   },
            '27': {'AO': [9],  # 腹主动脉
                   'CA': [21],  # 腹腔干
                   'LGA': [17],  # 胃左动脉
                   'SA': [20],  # 脾动脉
                   'RHA': [],  # 肝右动脉
                   'SMA': [8],  # 肠系膜上动脉
                   'other1': [16],
                   'other2': [23],
                   'other3': [],
                   'other4': [],
                   },
            '29a': {'AO': [4, 5, 6, 25],  # 腹主动脉
                    'CA': [24],  # 腹腔干
                    'LGA': [18],  # 胃左动脉
                    'SA': [23],  # 脾动脉
                    'RHA': [15],  # 肝右动脉
                    'SMA': [26],  # 肠系膜上动脉
                    'other1': [10],
                    'other2': [12],
                    'other3': [],
                    'other4': [],
                    },
            '29b': {'AO': [17],  # 腹主动脉
                    'CA': [14, 15],  # 腹腔干
                    'LGA': [13],  # 胃左动脉
                    'SA': [10],  # 脾动脉
                    'RHA': [1],  # 肝右动脉
                    'SMA': [28],  # 肠系膜上动脉
                    'other1': [16],
                    'other2': [23],
                    'other3': [20],
                    'other4': [],
                    },
            '30': {'AO': [40],  # 腹主动脉
                   'CA': [4, 39, 33],  # 腹腔干
                   'LGA': [31, 32],  # 胃左动脉
                   'SA': [25],  # 脾动脉
                   'RHA': [],  # 肝右动脉
                   'SMA': [38],  # 肠系膜上动脉
                   'other1': [3],
                   'other2': [45],
                   'other3': [],
                   'other4': [],
                   },
            '31': {'AO': [22],  # 腹主动脉
                   'CA': [21],  # 腹腔干
                   'LGA': [9],  # 胃左动脉
                   'SA': [20],  # 脾动脉
                   'RHA': [],  # 肝右动脉
                   'SMA': [8],  # 肠系膜上动脉
                   'other1': [3],
                   'other2': [23],
                   'other3': [],
                   'other4': [],
                   },
            '32': {'AO': [14],  # 腹主动脉
                   'CA': [9],  # 腹腔干
                   'LGA': [8],  # 胃左动脉
                   'SA': [7],  # 脾动脉
                   'RHA': [],  # 肝右动脉
                   'SMA': [16],  # 肠系膜上动脉
                   'other1': [13],
                   'other2': [11],
                   'other3': [],
                   'other4': [],
                   },
            '33': {'AO': [28, 29, 31, 32, 1],  # 腹主动脉
                   'CA': [27],  # 腹腔干
                   'LGA': [11],  # 胃左动脉
                   'SA': [26],  # 脾动脉
                   'RHA': [2],  # 肝右动脉
                   'SMA': [36],  # 肠系膜上动脉
                   'other1': [34],
                   'other2': [30],
                   'other3': [],
                   'other4': [],
                   },
            '35': {'AO': [25],  # 腹主动脉
                   'CA': [34],  # 腹腔干
                   'LGA': [28],  # 胃左动脉
                   'SA': [33],  # 脾动脉
                   'RHA': [],  # 肝右动脉
                   'SMA': [35, 36, 37],  # 肠系膜上动脉
                   'other1': [19],
                   'other2': [10],
                   'other3': [],
                   'other4': [],
                   },
            '36': {'AO': [22],  # 腹主动脉
                   'CA': [21],  # 腹腔干
                   'LGA': [20],  # 胃左动脉
                   'SA': [15],  # 脾动脉
                   'RHA': [],  # 肝右动脉
                   'SMA': [10],  # 肠系膜上动脉
                   'other1': [3],
                   'other2': [23],
                   'other3': [],
                   'other4': [],
                   },
            '38': {'AO': [3],  # 腹主动脉
                   'CA': [4, 9, 10],  # 腹腔干
                   'LGA': [7],  # 胃左动脉
                   'SA': [8],  # 脾动脉
                   'RHA': [],  # 肝右动脉
                   'SMA': [11, 16, 17],  # 肠系膜上动脉
                   'other1': [1],
                   'other2': [2],
                   'other3': [],
                   'other4': [],
                   },
            '39': {'AO': [22, 23, 24],  # 腹主动脉
                   'CA': [21],  # 腹腔干
                   'LGA': [20],  # 胃左动脉
                   'SA': [19],  # 脾动脉
                   'RHA': [],  # 肝右动脉
                   'SMA': [6],  # 肠系膜上动脉
                   'other1': [5],
                   'other2': [25],
                   'other3': [],
                   'other4': [],
                   },
            '40': {'AO': [22, 20, 49, 48, 50],  # 腹主动脉
                   'CA': [35],  # 腹腔干
                   'LGA': [31],  # 胃左动脉
                   'SA': [34],  # 脾动脉
                   'RHA': [],  # 肝右动脉
                   'SMA': [52],  # 肠系膜上动脉
                   'other1': [47],
                   'other2': [19],
                   'other3': [],
                   'other4': [],
                   },
            '41': {'AO': [33, 32],  # 腹主动脉
                   'CA': [25],  # 腹腔干
                   'LGA': [24],  # 胃左动脉
                   'SA': [23],  # 脾动脉
                   'RHA': [],  # 肝右动脉
                   'SMA': [31],  # 肠系膜上动脉
                   'other1': [34],
                   'other2': [8],
                   'other3': [],
                   'other4': [],
                   },
            '42': {'AO': [16, 1, 27],  # 腹主动脉
                   'CA': [24, 25, 26],  # 腹腔干
                   'LGA': [19],  # 胃左动脉
                   'SA': [22],  # 脾动脉
                   'RHA': [23],  # 肝右动脉
                   'SMA': [15],  # 肠系膜上动脉
                   'other1': [30],
                   'other2': [33],
                   'other3': [],
                   'other4': [],
                   },
            '43': {'AO': [22],  # 腹主动脉
                   'CA': [21, 20],  # 腹腔干
                   'LGA': [19],  # 胃左动脉
                   'SA': [15],  # 脾动脉
                   'RHA': [],  # 肝右动脉
                   'SMA': [6],  # 肠系膜上动脉
                   'other1': [9],
                   'other2': [23],
                   'other3': [],
                   'other4': [],
                   },
            '44': {'AO': [18],  # 腹主动脉
                   'CA': [15, 16, 17],  # 腹腔干
                   'LGA': [11],  # 胃左动脉
                   'SA': [12],  # 脾动脉
                   'RHA': [8],  # 肝右动脉
                   'SMA': [31],  # 肠系膜上动脉
                   'other1': [13],
                   'other2': [1],
                   'other3': [6],
                   'other4': [],
                   },
            '45': {'AO': [5],  # 腹主动脉
                   'CA': [16],  # 腹腔干
                   'LGA': [15],  # 胃左动脉
                   'SA': [14],  # 脾动脉
                   'RHA': [],  # 肝右动脉
                   'SMA': [32],  # 肠系膜上动脉
                   'other1': [22],
                   'other2': [27],
                   'other3': [29],
                   'other4': [],
                   },
            '46': {'AO': [1],  # 腹主动脉
                   'CA': [22],  # 腹腔干
                   'LGA': [18],  # 胃左动脉
                   'SA': [21],  # 脾动脉
                   'RHA': [],  # 肝右动脉
                   'SMA': [32],  # 肠系膜上动脉
                   'other1': [23],
                   'other2': [33],
                   'other3': [],
                   'other4': [],
                   },
            '47': {'AO': [21],  # 腹主动脉
                   'CA': [20],  # 腹腔干
                   'LGA': [12],  # 胃左动脉
                   'SA': [19],  # 脾动脉
                   'RHA': [],  # 肝右动脉
                   'SMA': [32],  # 肠系膜上动脉
                   'other1': [6],
                   'other2': [23],
                   'other3': [],
                   'other4': [],
                   },
            '48': {'AO': [4],  # 腹主动脉
                   'CA': [5, 6, 17],  # 腹腔干
                   'LGA': [12],  # 胃左动脉
                   'SA': [15],  # 脾动脉
                   'RHA': [16],  # 肝右动脉
                   'SMA': [21],  # 肠系膜上动脉
                   'other1': [3],
                   'other2': [7],
                   'other3': [],
                   'other4': [],
                   },
            '49': {'AO': [2],  # 腹主动脉
                   'CA': [7],  # 腹腔干
                   'LGA': [6],  # 胃左动脉
                   'SA': [5],  # 脾动脉
                   'RHA': [],  # 肝右动脉
                   'SMA': [12],  # 肠系膜上动脉
                   'other1': [1],
                   'other2': [8],
                   'other3': [],
                   'other4': [],
                   },
            '50': {'AO': [2, 7, 8],  # 腹主动脉
                   'CA': [22],  # 腹腔干
                   'LGA': [],  # 胃左动脉
                   'SA': [],  # 脾动脉
                   'RHA': [1],  # 肝右动脉
                   'SMA': [12],  # 肠系膜上动脉
                   'other1': [6],
                   'other2': [10],
                   'other3': [],
                   'other4': [],
                   },
            '52': {'AO': [2, 12, 8],  # 腹主动脉
                   'CA': [7],  # 腹腔干
                   'LGA': [6],  # 胃左动脉
                   'SA': [5],  # 脾动脉
                   'RHA': [],  # 肝右动脉
                   'SMA': [11],  # 肠系膜上动脉
                   'other1': [1],
                   'other2': [14],
                   'other3': [13],
                   'other4': [],
                   },
            '53': {'AO': [9],  # 腹主动脉
                   'CA': [26, 24],  # 腹腔干
                   'LGA': [20],  # 胃左动脉
                   'SA': [23],  # 脾动脉
                   'RHA': [25],  # 肝右动脉
                   'SMA': [29],  # 肠系膜上动脉
                   'other1': [8],
                   'other2': [5],
                   'other3': [28],
                   'other4': [],
                   },
            '55': {'AO': [6],  # 腹主动脉
                   'CA': [23, 24],  # 腹腔干
                   'LGA': [22],  # 胃左动脉
                   'SA': [13],  # 脾动脉
                   'RHA': [],  # 肝右动脉
                   'SMA': [39],  # 肠系膜上动脉
                   'other1': [3],
                   'other2': [4],
                   'other3': [],
                   'other4': [],
                   },
            '56': {'AO': [14, 28, 29],  # 腹主动脉
                   'CA': [9, 11, 12, 13],  # 腹腔干
                   'LGA': [5],  # 胃左动脉
                   'SA': [8],  # 脾动脉
                   'RHA': [10],  # 肝右动脉
                   'SMA': [26],  # 肠系膜上动脉
                   'other1': [27],
                   'other2': [18],
                   'other3': [],
                   'other4': [],
                   },
            '57': {'AO': [5],  # 腹主动脉
                   'CA': [10],  # 腹腔干
                   'LGA': [9],  # 胃左动脉
                   'SA': [6],  # 脾动脉
                   'RHA': [],  # 肝右动脉
                   'SMA': [3],  # 肠系膜上动脉
                   'other1': [14],
                   'other2': [20],
                   'other3': [4],
                   'other4': [],
                   },
            '58': {'AO': [10, 7, 6],  # 腹主动脉
                   'CA': [5],  # 腹腔干
                   'LGA': [4],  # 胃左动脉
                   'SA': [3],  # 脾动脉
                   'RHA': [],  # 肝右动脉
                   'SMA': [17],  # 肠系膜上动脉
                   'other1': [8],
                   'other2': [9],
                   'other3': [],
                   'other4': [],
                   },
            '59': {'AO': [1],  # 腹主动脉
                   'CA': [9],  # 腹腔干
                   'LGA': [7],  # 胃左动脉
                   'SA': [4],  # 脾动脉
                   'RHA': [8],  # 肝右动脉
                   'SMA': [22],  # 肠系膜上动脉
                   'other1': [11, 14],
                   'other2': [10],
                   'other3': [],
                   'other4': [],
                   },
            '60': {'AO': [4],  # 腹主动脉
                   'CA': [17],  # 腹腔干
                   'LGA': [12],  # 胃左动脉
                   'SA': [15],  # 脾动脉
                   'RHA': [16],  # 肝右动脉
                   'SMA': [34],  # 肠系膜上动脉
                   'other1': [5],
                   'other2': [3],
                   'other3': [35],
                   'other4': [],
                   },
            '61': {'AO': [20, 22, 23],  # 腹主动脉
                   'CA': [18, 19],  # 腹腔干
                   'LGA': [16],  # 胃左动脉
                   'SA': [17],  # 脾动脉
                   'RHA': [1],  # 肝右动脉
                   'SMA': [41],  # 肠系膜上动脉
                   'other1': [31],
                   'other2': [39],
                   'other3': [],
                   'other4': [],
                   },
            '62': {'AO': [27],  # 腹主动脉
                   'CA': [44, 43],  # 腹腔干
                   'LGA': [42],  # 胃左动脉
                   'SA': [33],  # 脾动脉
                   'RHA': [28],  # 肝右动脉
                   'SMA': [22],  # 肠系膜上动脉
                   'other1': [45],
                   'other2': [26],
                   'other3': [],
                   'other4': [],
                   },
            '63': {'AO': [22, 24, 26],  # 腹主动脉
                   'CA': [16, 21],  # 腹腔干
                   'LGA': [15],  # 胃左动脉
                   'SA': [9],  # 脾动脉
                   'RHA': [20],  # 肝右动脉
                   'SMA': [43],  # 肠系膜上动脉
                   'other1': [32],
                   'other2': [41],
                   'other3': [],
                   'other4': [],
                   },
            '64': {'AO': [17, 18, 19, 20],  # 腹主动脉
                   'CA': [16, 14],  # 腹腔干
                   'LGA': [13],  # 胃左动脉
                   'SA': [10],  # 脾动脉
                   'RHA': [15],  # 肝右动脉
                   'SMA': [9],  # 肠系膜上动脉
                   'other1': [4],
                   'other2': [3],
                   'other3': [],
                   'other4': [],
                   },
            '65': {'AO': [8],  # 腹主动脉
                   'CA': [22, 28],  # 腹腔干
                   'LGA': [11],  # 胃左动脉
                   'SA': [20],  # 脾动脉
                   'RHA': [21],  # 肝右动脉
                   'SMA': [7],  # 肠系膜上动脉
                   'other1': [27],
                   'other2': [31],
                   'other3': [32],
                   'other4': [],
                   },
            '66': {'AO': [9],  # 腹主动脉
                   'CA': [18],  # 腹腔干
                   'LGA': [17],  # 胃左动脉
                   'SA': [14],  # 脾动脉
                   'RHA': [],  # 肝右动脉
                   'SMA': [30],  # 肠系膜上动脉
                   'other1': [24],
                   'other2': [27],
                   'other3': [],
                   'other4': [],
                   },
            '67': {'AO': [4],  # 腹主动脉
                   'CA': [13, 14],  # 腹腔干
                   'LGA': [17],  # 胃左动脉
                   'SA': [22],  # 脾动脉
                   'RHA': [],  # 肝右动脉
                   'SMA': [12],  # 肠系膜上动脉
                   'other1': [11],
                   'other2': [9],
                   'other3': [],
                   'other4': [],
                   },
            '68': {'AO': [14],  # 腹主动脉
                   'CA': [32],  # 腹腔干
                   'LGA': [30],  # 胃左动脉
                   'SA': [21],  # 脾动脉
                   'RHA': [31],  # 肝右动脉
                   'SMA': [33],  # 肠系膜上动脉
                   'other1': [8],
                   'other2': [13],
                   'other3': [],
                   'other4': [],
                   },
            '69': {'AO': [1, 18],  # 腹主动脉
                   'CA': [17],  # 腹腔干
                   'LGA': [16],  # 胃左动脉
                   'SA': [4],  # 脾动脉
                   'RHA': [],  # 肝右动脉
                   'SMA': [29],  # 肠系膜上动脉
                   'other1': [30],
                   'other2': [37],
                   'other3': [],
                   'other4': [],
                   },
            '70': {'AO': [36],  # 腹主动脉
                   'CA': [23],  # 腹腔干
                   'LGA': [21],  # 胃左动脉
                   'SA': [22],  # 脾动脉
                   'RHA': [16],  # 肝右动脉
                   'SMA': [34, 35],  # 肠系膜上动脉
                   'other1': [8],
                   'other2': [15],
                   'other3': [],
                   'other4': [],
                   },
            '71': {'AO': [1, 2],  # 腹主动脉
                   'CA': [17, 19],  # 腹腔干
                   'LGA': [16],  # 胃左动脉
                   'SA': [10],  # 脾动脉
                   'RHA': [3],  # 肝右动脉
                   'SMA': [38],  # 肠系膜上动脉
                   'other1': [18],
                   'other2': [20],
                   'other3': [25],
                   'other4': [],
                   },
            '72': {'AO': [24],  # 腹主动脉
                   'CA': [23, 17],  # 腹腔干
                   'LGA': [11],  # 胃左动脉
                   'SA': [16, 14],  # 脾动脉
                   'RHA': [22],  # 肝右动脉
                   'SMA': [4],  # 肠系膜上动脉
                   'other1': [7],
                   'other2': [25],
                   'other3': [],
                   'other4': [],
                   },
            '73': {'AO': [13],  # 腹主动脉
                   'CA': [10],  # 腹腔干
                   'LGA': [9],  # 胃左动脉
                   'SA': [2],  # 脾动脉
                   'RHA': [],  # 肝右动脉
                   'SMA': [1],  # 肠系膜上动脉
                   'other1': [12],
                   'other2': [11],
                   'other3': [],
                   'other4': [],
                   },
            '74': {'AO': [48, 47, 45],  # 腹主动脉
                   'CA': [34],  # 腹腔干
                   'LGA': [44],  # 胃左动脉
                   'SA': [],  # 脾动脉
                   'RHA': [46],  # 肝右动脉
                   'SMA': [19],  # 肠系膜上动脉
                   'other1': [24],
                   'other2': [4],
                   'other3': [],
                   'other4': [],
                   },
            '78': {'AO': [32, 35, 33, 1, 31],  # 腹主动脉
                   'CA': [29, 30],  # 腹腔干
                   'LGA': [24],  # 胃左动脉
                   'SA': [28],  # 脾动脉
                   'RHA': [21],  # 肝右动脉
                   'SMA': [18],  # 肠系膜上动脉
                   'other1': [42],
                   'other2': [34],
                   'other3': [],
                   'other4': [],
                   },
            '79': {'AO': [36],  # 腹主动脉
                   'CA': [35],  # 腹腔干
                   'LGA': [34],  # 胃左动脉
                   'SA': [28],  # 脾动脉
                   'RHA': [],  # 肝右动脉
                   'SMA': [6],  # 肠系膜上动脉
                   'other1': [48],
                   'other2': [45],
                   'other3': [],
                   'other4': [],
                   },
            '80': {'AO': [29, 33, 34],  # 腹主动脉
                   'CA': [58, 59],  # 腹腔干
                   'LGA': [42],  # 胃左动脉
                   'SA': [57],  # 脾动脉
                   'RHA': [43],  # 肝右动脉
                   'SMA': [60],  # 肠系膜上动脉
                   'other1': [3],
                   'other2': [9],
                   'other3': [23],
                   'other4': [],
                   },
            '81': {'AO': [1, 9],  # 腹主动脉
                   'CA': [25, 29],  # 腹腔干
                   'LGA': [24],  # 胃左动脉
                   'SA': [19],  # 脾动脉
                   'RHA': [28],  # 肝右动脉
                   'SMA': [43],  # 肠系膜上动脉
                   'other1': [8],
                   'other2': [30],
                   'other3': [],
                   'other4': [],
                   },
            '82b': {'AO': [4, 14],  # 腹主动脉
                    'CA': [13, 11],  # 腹腔干
                    'LGA': [10],  # 胃左动脉
                    'SA': [7],  # 脾动脉
                    'RHA': [12],  # 肝右动脉
                    'SMA': [23],  # 肠系膜上动脉
                    'other1': [3],
                    'other2': [15],
                    'other3': [],
                    'other4': [],
                    },
            '83': {'AO': [2],  # 腹主动脉
                    'CA': [16],  # 腹腔干
                    'LGA': [15],  # 胃左动脉
                    'SA': [10],  # 脾动脉
                    'RHA': [],  # 肝右动脉
                    'SMA': [9],  # 肠系膜上动脉
                    'other1': [1],
                    'other2': [],
                    'other3': [],
                    'other4': [],
                    },
            '84': {'AO': [12],  # 腹主动脉
                   'CA': [27, 28],  # 腹腔干
                   'LGA': [46],  # 胃左动脉
                   'SA': [29],  # 脾动脉
                   'RHA': [35],  # 肝右动脉
                   'SMA': [26],  # 肠系膜上动脉
                   'other1': [17],
                   'other2': [18],
                   'other3': [],
                   'other4': [],
                   },
            '86': {'AO': [1],  # 腹主动脉
                   'CA': [9, 10],  # 腹腔干
                   'LGA': [8],  # 胃左动脉
                   'SA': [7],  # 脾动脉
                   'RHA': [2],  # 肝右动脉
                   'SMA': [19],  # 肠系膜上动脉
                   'other1': [15],
                   'other2': [17],
                   'other3': [],
                   'other4': [],
                   },
            '98': {'AO': [9],  # 腹主动脉
                   'CA': [36, 35, 34, 32, 29],  # 腹腔干
                   'LGA': [18],  # 胃左动脉
                   'SA': [28],  # 脾动脉
                   'RHA': [31],  # 肝右动脉
                   'SMA': [1],  # 肠系膜上动脉
                   'other1': [8],
                   'other2': [47],
                   'other3': [],
                   'other4': [],
                   },
            '101': {'AO': [16],  # 腹主动脉
                    'CA': [15],  # 腹腔干
                    'LGA': [14],  # 胃左动脉
                    'SA': [5],  # 脾动脉
                    'RHA': [],  # 肝右动脉
                    'SMA': [24],  # 肠系膜上动脉
                    'other1': [17],
                    'other2': [25],
                    'other3': [28],
                    'other4': [],
                    },
            '102': {'AO': [14, 20],  # 腹主动脉
                    'CA': [13],  # 腹腔干
                    'LGA': [11],  # 胃左动脉
                    'SA': [12],  # 脾动脉
                    'RHA': [],  # 肝右动脉
                    'SMA': [5],  # 肠系膜上动脉
                    'other1': [21],
                    'other2': [15],
                    'other3': [],
                    'other4': [],
                    },
            '103': {'AO': [35, 36, 90, 91, 92, 93, 94, 98, 73, 74, 75],  # 腹主动脉
                    'CA': [70, 72],  # 腹腔干
                    'LGA': [69],  # 胃左动脉
                    'SA': [47],  # 脾动脉
                    'RHA': [71],  # 肝右动脉
                    'SMA': [27],  # 肠系膜上动脉
                    'other1': [32],
                    'other2': [84],
                    'other3': [],
                    'other4': [],
                    },
            '104': {'AO': [3],  # 腹主动脉
                    'CA': [28, 20],  # 腹腔干
                    'LGA': [14],  # 胃左动脉
                    'SA': [19],  # 脾动脉
                    'RHA': [27],  # 肝右动脉
                    'SMA': [38],  # 肠系膜上动脉
                    'other1': [30],
                    'other2': [31],
                    'other3': [32],
                    'other4': [],
                    },
            '106': {'AO': [1, 24, 25, 26, 29, 30, 31, 33, 35, 36, 37, 38, 44, 45, 46],  # 腹主动脉
                    'CA': [21, 22, 23],  # 腹腔干
                    'LGA': [20],  # 胃左动脉
                    'SA': [2],  # 脾动脉
                    'RHA': [],  # 肝右动脉
                    'SMA': [32],  # 肠系膜上动脉
                    'other1': [34],
                    'other2': [],
                    'other3': [],
                    'other4': [],
                    },
            '107': {'AO': [4],  # 腹主动脉
                    'CA': [22],  # 腹腔干
                    'LGA': [20],  # 胃左动脉
                    'SA': [7],  # 脾动脉
                    'RHA': [17],  # 肝右动脉
                    'SMA': [8, 13, 17],  # 肠系膜上动脉
                    'other1': [11],
                    'other2': [3],
                    'other3': [],
                    'other4': [],
                    },
            '108': {'AO': [15, 16, 17, 18, 19, 20, 39],  # 腹主动脉
                    'CA': [38, 37],  # 腹腔干
                    'LGA': [36],  # 胃左动脉
                    'SA': [32],  # 脾动脉
                    'RHA': [21],  # 肝右动脉
                    'SMA': [53],  # 肠系膜上动脉
                    'other1': [7],
                    'other2': [14],
                    'other3': [],
                    'other4': [],
                    },
            '109': {'AO': [24],  # 腹主动脉
                    'CA': [17, 21, 23],  # 腹腔干
                    'LGA': [11],  # 胃左动脉
                    'SA': [16],  # 脾动脉
                    'RHA': [22],  # 肝右动脉
                    'SMA': [34],  # 肠系膜上动脉
                    'other1': [22],
                    'other2': [25],
                    'other3': [8],
                    'other4': [2],
                    },
            '110': {'AO': [11],  # 腹主动脉
                    'CA': [10],  # 腹腔干
                    'LGA': [2],  # 胃左动脉
                    'SA': [3, 8, 9],  # 脾动脉
                    'RHA': [],  # 肝右动脉
                    'SMA': [12],  # 肠系膜上动脉
                    'other1': [13],
                    'other2': [1],
                    'other3': [],
                    'other4': [],
                    },
            '111': {'AO': [12, 16],  # 腹主动脉
                    'CA': [17, 32, 33],  # 腹腔干
                    'LGA': [23],  # 胃左动脉
                    'SA': [31],  # 脾动脉
                    'RHA': [],  # 肝右动脉
                    'SMA': [1],  # 肠系膜上动脉
                    'other1': [11],
                    'other2': [40],
                    'other3': [],
                    'other4': [],
                    },
            '112': {'AO': [3],  # 腹主动脉
                    'CA': [6],  # 腹腔干
                    'LGA': [5],  # 胃左动脉
                    'SA': [4],  # 脾动脉
                    'RHA': [],  # 肝右动脉
                    'SMA': [2],  # 肠系膜上动脉
                    'other1': [1],
                    'other2': [7],
                    'other3': [],
                    'other4': [],
                    },
            '113': {'AO': [2],  # 腹主动脉
                    'CA': [],  # 腹腔干
                    'LGA': [],  # 胃左动脉
                    'SA': [],  # 脾动脉
                    'RHA': [],  # 肝右动脉
                    'SMA': [1],  # 肠系膜上动脉
                    'other1': [],
                    'other2': [],
                    'other3': [],
                    'other4': [],
                    },
            '114': {'AO': [18],  # 腹主动脉
                    'CA': [17, 17],  # 腹腔干
                    'LGA': [15],  # 胃左动脉
                    'SA': [9],  # 脾动脉
                    'RHA': [4],  # 肝右动脉
                    'SMA': [29],  # 肠系膜上动脉
                    'other1': [3],
                    'other2': [19],
                    'other3': [],
                    'other4': [],
                    },
            '116': {'AO': [18, 19],  # 腹主动脉
                    'CA': [9, 16, 17],  # 腹腔干
                    'LGA': [15],  # 胃左动脉
                    'SA': [10],  # 脾动脉
                    'RHA': [],  # 肝右动脉
                    'SMA': [8],  # 肠系膜上动脉
                    'other1': [5],
                    'other2': [1],
                    'other3': [],
                    'other4': [],
                    },
            '117': {'AO': [10, 8],  # 腹主动脉
                    'CA': [35],  # 腹腔干
                    'LGA': [34],  # 胃左动脉
                    'SA': [17],  # 脾动脉
                    'RHA': [],  # 肝右动脉
                    'SMA': [46],  # 肠系膜上动脉
                    'other1': [5],
                    'other2': [6],
                    'other3': [],
                    'other4': [],
                    },
            '118': {'AO': [8, 9],  # 腹主动脉
                    'CA': [19],  # 腹腔干
                    'LGA': [18],  # 胃左动脉
                    'SA': [12],  # 脾动脉
                    'RHA': [15],  # 肝右动脉
                    'SMA': [5],  # 肠系膜上动脉
                    'other1': [24],
                    'other2': [4],
                    'other3': [],
                    'other4': [],
                    },
            '121': {'AO': [2, 5, 6, 7, 10, 11],  # 腹主动脉
                    'CA': [17],  # 腹腔干
                    'LGA': [],  # 胃左动脉
                    'SA': [9],  # 脾动脉
                    'RHA': [15],  # 肝右动脉
                    'SMA': [14, 16],  # 肠系膜上动脉
                    'other1': [1],
                    'other2': [3],
                    'other3': [],
                    'other4': [],
                    },
            '122': {'AO': [4, 28],  # 腹主动脉
                    'CA': [7, 27],  # 腹腔干
                    'LGA': [26],  # 胃左动脉
                    'SA': [22],  # 脾动脉
                    'RHA': [23],  # 肝右动脉
                    'SMA': [12],  # 肠系膜上动脉
                    'other1': [5],
                    'other2': [3],
                    'other3': [6],
                    'other4': [],
                    },
            '123': {'AO': [26],  # 腹主动脉
                    'CA': [23, 24, 25],  # 腹腔干
                    'LGA': [16],  # 胃左动脉
                    'SA': [22],  # 脾动脉
                    'RHA': [17],  # 肝右动脉
                    'SMA': [27],  # 肠系膜上动脉
                    'other1': [5],
                    'other2': [10],
                    'other3': [],
                    'other4': [],
                    },
        }

    def get_contain(self, img_id):
        if img_id in self.contain_dict.keys():
            return self.contain_dict[img_id]
        else:
            logging.error('No contain info !')

            exit(-1)
