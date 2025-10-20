import os

import cc3d
import numpy as np
import xlrd2
from scipy import ndimage
from skimage import morphology
import tqdm

import toolkit_3D as tk3


def get_img_file_list(dataset_path):
    file_list = []
    img_name_list = os.listdir(dataset_path)
    for img_name in img_name_list:
        img_id = img_name.split(sep="_")[0]
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


def get_brief_info(brief_list, case_id, target):
    for brief_dict in brief_list:
        if brief_dict["case_id"] == int(case_id):
            if brief_dict["target"] == str(target):
                return brief_dict["wrapping"], brief_dict["deformation"]


def generate_contour_nii(file_dict, new_dir_path, prefix="contact_", contour_thickness=1.5, contact_range=2):
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

    # read nii image
    img_dict = tk3.get_nii(img_full_path)

    # get target contour
    contour_vein = tk3.get_3D_contour(img_dict["vein"], contour_thickness=contour_thickness)
    contour_artery = tk3.get_3D_contour(img_dict["artery"], contour_thickness=contour_thickness)

    # get target-tumor contact
    tumor = img_dict["tumor"]
    contact_vein = tk3.get_3D_contact(tumor, contour_vein, contact_range=contact_range)
    contact_artery = tk3.get_3D_contact(tumor, contour_artery, contact_range=contact_range)

    new_data = tumor * 1 + contour_vein * 2 + contact_vein + contour_artery * 4 + contact_artery

    file_dict["img_contact_path"] = new_full_path

    tk3.save_nii(new_data, new_full_path, img_dict["info"])


def calculate_2D_contact(file_dict, target, size_threshold=0):
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
    # img_contact_dict = get_contour_img(contact_img_path)
    img_contact_dict = tk3.get_contour_nii(contact_img_path)
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
            # contact_px_num = 0
            # for i in range(0, img_contour_slice.shape[0]):
            #     for j in range(0, img_contour_slice.shape[1]):
            #         if img_contact_slice[i, j] == 1 and img_sep_contour_slice[i, j] == 1:
            #             contact_px_num += 1
            img_sep_contact_slice = np.multiply(img_sep_contour_slice, img_contact_slice)
            contact_px_num = np.sum(img_sep_contact_slice[img_sep_contact_slice == 1])

            if contact_px_num == 0 or contour_px_num == 0:
                ratio = 0
            else:
                ratio = contact_px_num / contour_px_num

            if ratio <= min_ratio:
                continue

            part_result_dict = {
                "contour_px_num": contour_px_num,
                "contact_px_num": contact_px_num,
                "ratio": ratio
            }
            part_result_list.append(part_result_dict)
            if ratio > max_ratio:
                max_ratio = ratio
                max_slice = slice_num
                max_ratio_contour = contour_px_num
                max_ratio_contact = contact_px_num

            if ratio > slice_max_ratio:
                slice_max_ratio = ratio

        # if the slice has no ratio greater than min_ratio, this slice is un-contacted
        # do not save this empty slice
        if slice_max_ratio <= min_ratio:
            continue

        slice_result_dict = {
            "slice_num": slice_num,
            "slice_result_list": part_result_list,
            "slice_max_ratio": slice_max_ratio,
            "slice_max_ratio_contour": max_ratio_contour,
            "slice_max_ratio_contact": max_ratio_contact
        }
        slice_result_list.append(slice_result_dict)

    # if every slice is empty slice, mark the case with slice_num = -1
    if len(slice_result_list) == 0:
        slice_result_dict = {
            "slice_num": -1,
            "slice_result_list": [],
            "slice_max_ratio": 0,
            "slice_max_ratio_contour": 0,
            "slice_max_ratio_contact": 0
        }
        slice_result_list.append(slice_result_dict)
    return slice_result_list, max_ratio, max_slice


def thin_detect(part_img, path_list, img_tumor, thin_threshold=2, tumor_distance=5, isl_max_num=3, img_info=None, target="vein"):
    """


    :param tumor_distance:
    :param part_img: the output of [skele.distance_to_category]
    :param path_list:
    :param thin_threshold:
    :param isl_max_num: the maximum of island number, such as (2, 3), None for no restrain
    :return:
    """

    def thin_locate(part_erode, part_eroded):
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
                minus_map = np.where(minus_map < 0.5, 1, 0)
                minus_map = np.multiply(minus_map, part_eroded)
                minus_map_total += minus_map
        if np.sum(minus_map_total[minus_map_total > 0]) > 0:
            result = np.where(minus_map_total > 0, 0, 1)
            return result
        else:
            return None

    part_id_range = range(1, int(np.max(part_img)) + 1)
    result_img = np.zeros(part_img.shape)
    result_info_list = []

    for idx, part_id in enumerate(tqdm.tqdm(part_id_range, desc="thin detect")):
        # for part_id in part_id_range:
        # For every vessel part in the [part_img]
        part = np.where(part_img == part_id, 1, 0)
        thin_degree = 0
        has_thin = False
        thin_locate_result = None

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
            has_thin = True
            thin_locate_result = thin_locate(part_erode, part_eroded)

            break

        if has_thin:
            if thin_locate_result is None:
                continue
            distance_map = ndimage.distance_transform_edt(thin_locate_result)

            distance_map = np.multiply(distance_map, img_tumor)

            # tk3.save_nii(np.where(thin_locate_result == 0, 1, 0) + part, "suspect thin/part_" + str(part_id) +
            #              "_distance.nii.gz", img_info)

            distance_map = np.where(distance_map == 0, tumor_distance + 1, distance_map)

            if np.sum(distance_map[distance_map <= tumor_distance]) > 0:
                # result_img += np.where(part_img == part_id, part_img, 0)
                if target == "vein":
                    result_img += np.where(part_img == part_id, 3, 0)
                    result_img += np.where(thin_locate_result == 0, 2, 0)
                elif target == "artery":
                    result_img += np.where(part_img == part_id, 1, 0)
                    result_img += np.where(thin_locate_result == 0, 3, 0)
                result_info_dict = {
                    "part_id": part_id,
                    "thin_degree": thin_degree,
                    "part_radis": path_list[part_id - 1]["weight"]
                }
                result_info_list.append(result_info_dict)

    if np.sum(result_img[result_img > 0]) > 0:
        return result_img, result_info_list
    else:
        return None, None


# def thin_detect(part_img, path_list, img_tumor, thin_threshold=2, tumor_distance=5, isl_max_num=3, img_info=None):
#     """
#
#
#     :param tumor_distance:
#     :param part_img: the output of [skele.distance_to_category]
#     :param path_list:
#     :param thin_threshold:
#     :param isl_max_num: the maximum of island number, such as (2, 3), None for no restrain
#     :return:
#     """
#
#     def thin_locate(part_erode, part_eroded):
#         # kernel = morphology.ball(1)
#         # img_dilation = morphology.dilation(image, kernel)
#         n, islands = tk3.get_islands_num(part_erode)
#         dist_map_list = []
#         minus_map_total = np.zeros(part_erode.shape)
#         for i in range(1, n + 1):
#             dist_map = np.where(islands == i, 0, 1)
#             dist_map = ndimage.distance_transform_edt(dist_map)
#             dist_map = np.multiply(dist_map, part_eroded)
#             dist_map_list.append(dist_map)
#             if i == 1:
#                 continue
#             for j in range(0, i - 1):
#                 minus_map = np.abs(dist_map_list[j] - dist_map)
#                 minus_map = np.where(minus_map < 0.5, 1, 0)
#                 minus_map = np.multiply(minus_map, part_eroded)
#                 minus_map_total += minus_map
#         if np.sum(minus_map_total[minus_map_total > 0]) > 0:
#             result = np.where(minus_map_total > 0, 1, 0)
#             return result
#         else:
#             return None
#
#     part_id_range = range(1, int(np.max(part_img)) + 1)
#     result_img = np.zeros(part_img.shape)
#
#     for idx, part_id in enumerate(tqdm.tqdm(part_id_range, desc="thin detect")):
#     # for part_id in part_id_range:
#         # For every vessel part in the [part_img]
#         part = np.where(part_img == part_id, 1, 0)
#         thin_point_list = []
#         thin_degree = 0
#         has_thin = False
#
#         # tumor_distance = path_list[part_id - 1]["weight"] + 3
#
#         # tk3.save_nii(part, "suspect thin/part_" + str(part_id) + "_erode_" + str(0) + ".nii.gz", img_info)
#         for thin_times in range(1, thin_threshold + 1):
#             # Erode the part from outside layer by layer
#             part_erode, part_eroded = tk3.erode(part, thin_times)
#             # tk3.save_nii(part_erode, "suspect thin/part_" + str(part_id) + "_erode_" + str(thin_times) + ".nii.gz", img_info)
#
#             # Exclude too small vessels and extreme condition
#             # Restrain the broking into islands number
#             isl_num, isl_img = tk3.get_islands_num(part_erode)
#             print("    " + str(part_id) + " - isl_num = " + str(isl_num))
#             if isl_num == 0:
#                 break
#             if isl_num == 1:
#                 continue
#             if isl_max_num is not None:
#                 if isl_num > isl_max_num:
#                     break
#             # Once the one connect part is eroded into several parts
#             thin_degree = thin_times
#             has_thin = True
#
#             path_point_list = path_list[part_id - 1]["path"]
#             path_start = 0
#             path_end = len(path_point_list)
#             path_range = list(range(0, len(path_point_list)))
#
#             # Clear the path point in the end of the part's path
#             for i in path_range:
#                 if part_erode[path_point_list[i]] == 0:
#                     path_start += 1
#                 else:
#                     break
#             for i in path_range[::-1]:
#                 if part_erode[path_point_list[i]] == 0:
#                     path_end -= 1
#                 else:
#                     break
#
#             # Find the path point in the eroded break position
#             for i in range(path_start, path_end):
#                 path_point = path_point_list[i]
#
#                 if part_eroded[path_point] == 1:
#                     thin_point_list.append(path_point)
#             break
#
#         if has_thin:
#             distance_map = np.ones(part_img.shape)
#             for thin_point in thin_point_list:
#                 distance_map[thin_point] = 0
#             distance_map = ndimage.distance_transform_edt(distance_map)
#
#             distance_map = np.multiply(distance_map, img_tumor)
#
#             # tk3.save_nii(distance_map + part_erode, "suspect thin/part_" + str(part_id) + "_distance" + ".nii.gz", img_info)
#
#             distance_map = np.where(distance_map == 0, tumor_distance + 1, distance_map)
#
#             if np.sum(distance_map[distance_map <= tumor_distance]) > 0:
#                 result_img += np.where(part_img == part_id, part_img, 0)
#
#     if np.sum(result_img[result_img > 0]) > 0:
#         return result_img
#     else:
#         return None
