import sys

import nibabel as nib
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt



sys.setrecursionlimit(500000)


def get_nii_slices(filename, sliceNum):
    # read nii 3D data
    nii = nib.load(filename)
    img = nii.get_fdata().astype(np.float32)

    # get 2D slices
    img = img[:, :, sliceNum]

    # rotate 90 to fit itk-snap
    img = np.rot90(img, k=1, axes=(0, 1))

    # seperate labels
    img_background = np.where(img == 0, 1, 0).astype(np.float32)
    img_artery = np.where(img == 1, 1, 0).astype(np.float32)
    img_tumor = np.where(img == 2, 1, 0).astype(np.float32)
    img_vein = np.where(img == 3, 1, 0).astype(np.float32)

    result_dict = {
        "origin": img.astype(np.float32),
        "background": img_background,
        "artery": img_artery,
        "tumor": img_tumor,
        "vein": img_vein
    }

    return result_dict


def contours_detect(array):
    contours = np.zeros(array.shape)
    indices = np.where(array == 1)
    threshold = 5
    for i, j in zip(list(indices[0]), list(indices[1])):
        if sum_surrounding(array, i, j) < threshold:
            contours[i, j] = 1
    return contours


def contours_detect_canny(array):
    contours_vein = cv.Canny(array.astype(np.uint8), -1, -1)
    contours_vein = np.where(contours_vein > 0, 1, 0)

    ax = plt.matshow(contours_vein)
    plt.colorbar(ax.colorbar, fraction=0.025)
    plt.show()

    return contours_vein


def expand(array, level):
    expand_img = np.array(array, copy=True)
    indices = np.where(expand_img == 1)
    array_height = array.shape[0]
    array_width = array.shape[1]
    for i, j in zip(list(indices[0]), list(indices[1])):
        for (a, b) in get_level_list(level):
            if i + a > array_height - 1 or i + a < 0 or j + b > array_width - 1 or j + b < 0:
                continue
            expand_img[i + a][j + b] = 1
    return expand_img


def get_level_list(level):
    # level_list = []
    # for i in range(- level, level + 1):
    #     for j in range(- level, level):
    #         level_list.append((i, j))
    # return level_list

    level_list = []
    if level == 1:
        level_list = [(-1, 0), (1, 0), (0, 1), (0, -1),
                      (-1, -1), (1, -1), (-1, 1), (1, 1)]
        # [X][X][X]
        # [X][O][X]
        # [X][X][X]
    elif level == 2:
        level_list = [(-2, 1), (-2, 0), (-2, -1),
                      (-1, 2), (-1, 1), (-1, 0), (-1, -1), (-1, -2),
                      (0, 2), (0, 1), (0, -1), (0, -2),
                      (1, 2), (1, 1), (1, 0), (1, -1), (1, -2),
                      (2, 1), (2, 0), (2, -1)]
        # [ ][X][X][X][ ]
        # [X][X][X][X][X]
        # [X][X][O][X][X]
        # [X][X][X][X][X]
        # [ ][X][X][X][ ]
    return level_list


def sum_surrounding(array, x, y):
    array_height = array.shape[0]
    array_width = array.shape[1]

    # sum = get_located_pixel(array, array_width, array_height, x - 1, y - 1)
    sum = get_located_pixel(array, array_width, array_height, x, y - 1)
    # sum += get_located_pixel(array, array_width, array_height, x + 1, y - 1)
    sum += get_located_pixel(array, array_width, array_height, x - 1, y)
    sum += get_located_pixel(array, array_width, array_height, x, y)
    sum += get_located_pixel(array, array_width, array_height, x + 1, y)
    # sum += get_located_pixel(array, array_width, array_height, x - 1, y + 1)
    sum += get_located_pixel(array, array_width, array_height, x, y + 1)
    # sum += get_located_pixel(array, array_width, array_height, x + 1, y + 1)

    return sum


def get_located_pixel(array, width, height, x, y):
    if x < 0 or x >= width:
        return 0
    elif y < 0 or y >= height:
        return 0
    else:
        return array[x, y]


def get_contact_related_contours(img, contact):
    def mark(img, result, x, y):
        if img[x, y] == result[x, y]:
            return
        else:
            result[x, y] = 1
            if x > 0:
                mark(img, result, x - 1, y)
            if y > 0:
                mark(img, result, x, y - 1)
            if x < img.shape[1] - 1:
                mark(img, result, x + 1, y)
            if y < img.shape[0] - 1:
                mark(img, result, x, y + 1)
            if x > 0 and y > 0:
                mark(img, result, x - 1, y - 1)
            if x < img.shape[1] - 1 and y < img.shape[0] - 1:
                mark(img, result, x + 1, y + 1)
            if x > 0 and y < img.shape[0] - 1:
                mark(img, result, x + 1, y - 1)
            if y > 0 and x < img.shape[1] - 1:
                mark(img, result, x - 1, y + 1)

    result = np.zeros(contact.shape)

    indices = np.where(contact == 1)
    for i, j in zip(list(indices[0]), list(indices[1])):
        mark(img, result, i, j)
    return result


def show_result(img, contact, contours, filename, sliceNum, target, show_img=False):
    """
    0 - 背景
    1 - 动脉
    2 - 肿瘤
    3 - 静脉
    8 - 静脉边界
    13 - 静脉-肿瘤接触
    """

    # 计算接触/周长比例
    total_contact_contours = np.sum(contact[contact == 1])
    total_contours = np.sum(contours[contours == 1])
    contact_ratio = total_contact_contours / total_contours

    if total_contact_contours > 0:
        if show_img:
            result_img = img.copy()
            result_img += contours * 20
            result_img -= contact * 7
            ax = plt.matshow(result_img)
            plt.colorbar(ax.colorbar, fraction=0.025)
            plt.title("z-slice {}".format(sliceNum + 1) + " in " + filename + "\n" + target + "-tumor contact ratio: {:.2%}".format(contact_ratio))
            plt.show()
        print("[" + filename + "] s" + str(sliceNum) + " " + target + "-tumor contact ratio: " + str(total_contact_contours) + " / " + str(total_contours) + " = {:.2%}".format(contact_ratio))

    return contact_ratio


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


def calculate_2D_contact(file_dict, target, size_threshold=0, print_info=True):
    min_contour_px_num = 5
    min_ratio = 0.1

    contact_img_path = file_dict["img_contact_path"]
    img_contact_dict = get_contour_img(contact_img_path)
    # background - 0
    # tumor - 1
    # contour_vein - 2
    # contact_vein - 3
    # contour_artery - 4
    # contact_artery - 5
    img_contour = img_contact_dict["contour_" + target]
    img_contact = img_contact_dict["contact_" + target]

    total_slice = img_contour.shape[0]

    result_list = []
    max_ratio = 0
    max_slice = -1

    for slice_num in range(0, total_slice):
        # For every slice
        img_contour_slice = img_contour[slice_num, :, :]
        img_contact_slice = img_contact[slice_num, :, :]
        sep_target_list = get_sep_list(img_contour_slice, size_threshold)

        slice_result_list = []
        slice_max_ratio = 0
        max_ratio_contour = 0
        max_ratio_contact = 0

        for img_sep_contour_slice in sep_target_list:
            # For every part
            contour_px_num = np.sum(img_sep_contour_slice[img_sep_contour_slice == 1])
            if contour_px_num <= min_contour_px_num:
                continue
            contact_px_num = 0
            for i in range(0, img_contour_slice.shape[0]):
                for j in range(0, img_contour_slice.shape[1]):
                    if img_contact_slice[i, j] == 1 and img_sep_contour_slice[i, j] == 1:
                        contact_px_num += 1

            if contact_px_num == 0 or contour_px_num == 0:
                ratio = 0
            else:
                ratio = contact_px_num / contour_px_num

            if ratio <= min_ratio:
                continue

            if print_info:
                print("contour pixel: " + str(contour_px_num) + "    contact pixel: " + str(
                    contact_px_num) + "    ratio: " + str(ratio))

            slice_result_dict = {
                "contour_px_num": contour_px_num,
                "contact_px_num": contact_px_num,
                "ratio": ratio
            }
            slice_result_list.append(slice_result_dict)
            if ratio > max_ratio:
                max_ratio = ratio
                max_slice = slice_num
                max_ratio_contour = contour_px_num
                max_ratio_contact = contact_px_num

            if ratio > slice_max_ratio:
                slice_max_ratio = ratio

        if slice_max_ratio <= min_ratio:
            continue
        result_dict = {
            "slice_num": slice_num,
            "slice_result_list": slice_result_list,
            "slice_max_ratio": slice_max_ratio,
            "slice_max_ratio_contour": max_ratio_contour,
            "slice_max_ratio_contact": max_ratio_contact
        }
        result_list.append(result_dict)
    if len(result_list) == 0:
        result_dict = {
            "slice_num": -1,
            "slice_result_list": [],
            "slice_max_ratio": 0,
            "slice_max_ratio_contour": 0,
            "slice_max_ratio_contact": 0
        }
        result_list.append(result_dict)
    return result_list, max_ratio, max_slice


if __name__ == '__main__':

    filename = "30_seg.nii.gz"
    # filename = "32_seg.nii.gz"
    # filename = "35_seg.nii.gz"

    sliceNum = 14

    target = "vein"
    # target = "artery"

    expand_level = 2

    if sliceNum > 0:
        show_img = True

        # sliceNum -= 1

        img2D_dict = get_nii_slices(filename, sliceNum)

        img = img2D_dict["origin"]
        img_target = img2D_dict[target]
        img_tumor = img2D_dict["tumor"]

        vein_list = get_sep_list(img_target)

        for vein in vein_list:
            # get vein contours
            # contours_vein = contours_detect_canny(vein).astype(np.float32)
            contours_vein = contours_detect(vein).astype(np.float32)

            # get expanded tumor
            expand_tumor = expand(img_tumor, expand_level).astype(np.float32)

            # ax = plt.matshow(contours_vein)
            # plt.colorbar(ax.colorbar, fraction=0.025)
            # plt.show()


            # calculate the superposition of vein contours and expanded tumor
            contact = np.multiply(contours_vein, expand_tumor)

            # calculate contact / total contours and show img
            contact_ratio = show_result(img, contact, contours_vein, filename, sliceNum, target, show_img)

    else:
        show_img = False
        for sliceNum in range(0, 63):

            img2D_dict = get_nii_slices(filename, sliceNum)

            img = img2D_dict["origin"]
            img_target = img2D_dict[target]
            img_tumor = img2D_dict["tumor"]

            vein_list = get_sep_list(img_target)

            for vein in vein_list:
                # get vein contours
                # contours_vein = contours_detect_canny(vein).astype(np.float32)
                contours_vein = contours_detect(vein).astype(np.float32)

                # get expanded tumor
                expand_tumor = expand(img_tumor, expand_level).astype(np.float32)

                # ax = plt.matshow(contours_vein)
                # plt.colorbar(ax.colorbar, fraction=0.025)
                # plt.show()

                # calculate the superposition of vein contours and expanded tumor
                contact = np.multiply(contours_vein, expand_tumor)

                # calculate contact / total contours and show img
                contact_ratio = show_result(img, contact, contours_vein, filename, sliceNum, target)
"""
3D 处理想法：
1. 在 3D 环境下处理，直接分离出分立的血管，按血管保留计算接触
2. 研究 3D 环境下的边界处理和交界面处理
3. 在一定的轴向区域中考虑比例问题

"""

# [30_seg.nii.gz] s15 artery-tumor contact ratio: 19.05%
# [30_seg.nii.gz] s16 artery-tumor contact ratio: 38.10%
# [30_seg.nii.gz] s17 artery-tumor contact ratio: 9.09%
# [30_seg.nii.gz] s19 artery-tumor contact ratio: 29.17%
# [30_seg.nii.gz] s20 artery-tumor contact ratio: 7.69%
# [30_seg.nii.gz] s21 artery-tumor contact ratio: 35.71%
# [30_seg.nii.gz] s22 artery-tumor contact ratio: 31.03%
# [30_seg.nii.gz] s23 artery-tumor contact ratio: 41.38%
# [30_seg.nii.gz] s24 artery-tumor contact ratio: 18.52%
# [30_seg.nii.gz] s25 artery-tumor contact ratio: 29.63%
# [30_seg.nii.gz] s26 artery-tumor contact ratio: 15.38%
# [30_seg.nii.gz] s27 artery-tumor contact ratio: 12.50%
# [30_seg.nii.gz] s28 artery-tumor contact ratio: 42.86%
# [30_seg.nii.gz] s29 artery-tumor contact ratio: 30.43%
# [30_seg.nii.gz] s30 artery-tumor contact ratio: 16.67%
# [30_seg.nii.gz] s31 artery-tumor contact ratio: 19.23%
# [30_seg.nii.gz] s32 artery-tumor contact ratio: 14.29%
# [30_seg.nii.gz] s33 artery-tumor contact ratio: 10.34%

# [30_seg.nii.gz] s13 vein-tumor contact ratio: 2.00%
# [30_seg.nii.gz] s14 vein-tumor contact ratio: 27.66%
# [30_seg.nii.gz] s14 vein-tumor contact ratio: 61.54%
# [30_seg.nii.gz] s15 vein-tumor contact ratio: 34.69%
# [30_seg.nii.gz] s16 vein-tumor contact ratio: 24.00%
# [30_seg.nii.gz] s17 vein-tumor contact ratio: 13.21%
# [30_seg.nii.gz] s18 vein-tumor contact ratio: 17.86%
# [30_seg.nii.gz] s19 vein-tumor contact ratio: 18.64%
# [30_seg.nii.gz] s20 vein-tumor contact ratio: 26.09%
# [30_seg.nii.gz] s21 vein-tumor contact ratio: 14.47%
# [30_seg.nii.gz] s22 vein-tumor contact ratio: 13.95%
# [30_seg.nii.gz] s23 vein-tumor contact ratio: 20.75%
# [30_seg.nii.gz] s24 vein-tumor contact ratio: 16.98%
# [30_seg.nii.gz] s25 vein-tumor contact ratio: 15.91%
# [30_seg.nii.gz] s26 vein-tumor contact ratio: 5.26%


# [32_seg.nii.gz] s45 artery-tumor contact ratio: 12.68%
# [32_seg.nii.gz] s45 artery-tumor contact ratio: 8.82%
# [32_seg.nii.gz] s46 artery-tumor contact ratio: 14.47%
# [32_seg.nii.gz] s47 artery-tumor contact ratio: 82.22%
# [32_seg.nii.gz] s47 artery-tumor contact ratio: 12.37%
# [32_seg.nii.gz] s48 artery-tumor contact ratio: 90.91%
# [32_seg.nii.gz] s48 artery-tumor contact ratio: 22.00%
# [32_seg.nii.gz] s49 artery-tumor contact ratio: 24.27%
# [32_seg.nii.gz] s50 artery-tumor contact ratio: 39.29%
# [32_seg.nii.gz] s50 artery-tumor contact ratio: 2.38%
# [32_seg.nii.gz] s51 artery-tumor contact ratio: 40.74%
# [32_seg.nii.gz] s51 artery-tumor contact ratio: 14.29%
# [32_seg.nii.gz] s52 artery-tumor contact ratio: 50.00%
# [32_seg.nii.gz] s52 artery-tumor contact ratio: 100.00%
# [32_seg.nii.gz] s52 artery-tumor contact ratio: 100.00%
# [32_seg.nii.gz] s53 artery-tumor contact ratio: 68.18%
# [32_seg.nii.gz] s54 artery-tumor contact ratio: 85.71%
# [32_seg.nii.gz] s54 artery-tumor contact ratio: 11.84%
# [32_seg.nii.gz] s55 artery-tumor contact ratio: 100.00%
# [32_seg.nii.gz] s55 artery-tumor contact ratio: 95.45%
# [32_seg.nii.gz] s55 artery-tumor contact ratio: 2.60%
# [32_seg.nii.gz] s56 artery-tumor contact ratio: 100.00%
# [32_seg.nii.gz] s56 artery-tumor contact ratio: 96.00%
# [32_seg.nii.gz] s57 artery-tumor contact ratio: 90.12%
# [32_seg.nii.gz] s58 artery-tumor contact ratio: 90.00%
# [32_seg.nii.gz] s58 artery-tumor contact ratio: 80.00%
# [32_seg.nii.gz] s58 artery-tumor contact ratio: 60.87%
# [32_seg.nii.gz] s59 artery-tumor contact ratio: 88.89%
# [32_seg.nii.gz] s60 artery-tumor contact ratio: 83.33%
# [32_seg.nii.gz] s61 artery-tumor contact ratio: 69.23%
# [32_seg.nii.gz] s62 artery-tumor contact ratio: 42.11%
# [32_seg.nii.gz] s47 vein-tumor contact ratio: 24.00%
# [32_seg.nii.gz] s48 vein-tumor contact ratio: 13.73%
# [32_seg.nii.gz] s49 vein-tumor contact ratio: 19.57%
# [32_seg.nii.gz] s50 vein-tumor contact ratio: 26.19%
# [32_seg.nii.gz] s51 vein-tumor contact ratio: 29.27%
# [32_seg.nii.gz] s52 vein-tumor contact ratio: 31.71%
# [32_seg.nii.gz] s53 vein-tumor contact ratio: 25.53%
# [32_seg.nii.gz] s54 vein-tumor contact ratio: 27.66%
# [32_seg.nii.gz] s55 vein-tumor contact ratio: 24.53%
# [32_seg.nii.gz] s56 vein-tumor contact ratio: 24.56%
# [32_seg.nii.gz] s57 vein-tumor contact ratio: 15.69%
# [32_seg.nii.gz] s58 vein-tumor contact ratio: 16.07%
# [32_seg.nii.gz] s59 vein-tumor contact ratio: 11.76%
# [32_seg.nii.gz] s60 vein-tumor contact ratio: 9.86%


# [35_seg.nii.gz] s28 artery-tumor contact ratio: 26.92%
# [35_seg.nii.gz] s29 artery-tumor contact ratio: 20.00%
# [35_seg.nii.gz] s30 artery-tumor contact ratio: 16.67%
# [35_seg.nii.gz] s32 artery-tumor contact ratio: 20.00%
# [35_seg.nii.gz] s33 artery-tumor contact ratio: 60.00%
# [35_seg.nii.gz] s34 artery-tumor contact ratio: 60.00%
# [35_seg.nii.gz] s35 artery-tumor contact ratio: 75.00%
# [35_seg.nii.gz] s41 artery-tumor contact ratio: 28.57%
# [35_seg.nii.gz] s42 artery-tumor contact ratio: 47.37%
# [35_seg.nii.gz] s43 artery-tumor contact ratio: 66.67%
# [35_seg.nii.gz] s43 artery-tumor contact ratio: 50.00%
# [35_seg.nii.gz] s44 artery-tumor contact ratio: 22.22%
# [35_seg.nii.gz] s44 artery-tumor contact ratio: 21.05%
# [35_seg.nii.gz] s45 artery-tumor contact ratio: 33.33%
# [35_seg.nii.gz] s45 artery-tumor contact ratio: 11.76%
# [35_seg.nii.gz] s46 artery-tumor contact ratio: 9.52%
# [35_seg.nii.gz] s47 artery-tumor contact ratio: 100.00%
# [35_seg.nii.gz] s47 artery-tumor contact ratio: 10.00%
# [35_seg.nii.gz] s48 artery-tumor contact ratio: 100.00%
# [35_seg.nii.gz] s48 artery-tumor contact ratio: 15.00%
# [35_seg.nii.gz] s49 artery-tumor contact ratio: 66.67%
# [35_seg.nii.gz] s49 artery-tumor contact ratio: 15.38%
# [35_seg.nii.gz] s50 artery-tumor contact ratio: 100.00%
# [35_seg.nii.gz] s50 artery-tumor contact ratio: 20.00%
# [35_seg.nii.gz] s51 artery-tumor contact ratio: 100.00%
# [35_seg.nii.gz] s51 artery-tumor contact ratio: 20.79%
# [35_seg.nii.gz] s52 artery-tumor contact ratio: 100.00%
# [35_seg.nii.gz] s52 artery-tumor contact ratio: 7.61%
# [35_seg.nii.gz] s53 artery-tumor contact ratio: 100.00%
# [35_seg.nii.gz] s54 artery-tumor contact ratio: 100.00%
# [35_seg.nii.gz] s55 artery-tumor contact ratio: 62.50%
# [35_seg.nii.gz] s55 artery-tumor contact ratio: 100.00%
# [35_seg.nii.gz] s56 artery-tumor contact ratio: 91.89%
# [35_seg.nii.gz] s56 artery-tumor contact ratio: 100.00%
# [35_seg.nii.gz] s57 artery-tumor contact ratio: 85.88%
# [35_seg.nii.gz] s57 artery-tumor contact ratio: 50.00%
# [35_seg.nii.gz] s58 artery-tumor contact ratio: 28.57%
# [35_seg.nii.gz] s58 artery-tumor contact ratio: 69.84%
# [35_seg.nii.gz] s59 artery-tumor contact ratio: 33.33%
# [35_seg.nii.gz] s59 artery-tumor contact ratio: 46.48%
# [35_seg.nii.gz] s60 artery-tumor contact ratio: 40.18%
# [35_seg.nii.gz] s61 artery-tumor contact ratio: 29.41%
# [35_seg.nii.gz] s61 artery-tumor contact ratio: 54.05%
# [35_seg.nii.gz] s62 artery-tumor contact ratio: 2.38%
# [35_seg.nii.gz] s62 artery-tumor contact ratio: 43.59%

# [35_seg.nii.gz] s37 vein-tumor contact ratio: 7.55%
# [35_seg.nii.gz] s38 vein-tumor contact ratio: 9.80%
# [35_seg.nii.gz] s39 vein-tumor contact ratio: 16.67%
# [35_seg.nii.gz] s40 vein-tumor contact ratio: 22.73%
# [35_seg.nii.gz] s41 vein-tumor contact ratio: 35.90%
# [35_seg.nii.gz] s42 vein-tumor contact ratio: 36.11%
# [35_seg.nii.gz] s43 vein-tumor contact ratio: 45.00%
# [35_seg.nii.gz] s44 vein-tumor contact ratio: 28.57%
# [35_seg.nii.gz] s45 vein-tumor contact ratio: 18.75%
# [35_seg.nii.gz] s46 vein-tumor contact ratio: 16.13%
# [35_seg.nii.gz] s47 vein-tumor contact ratio: 33.33%
# [35_seg.nii.gz] s48 vein-tumor contact ratio: 15.79%
# [35_seg.nii.gz] s49 vein-tumor contact ratio: 8.11%
# [35_seg.nii.gz] s50 vein-tumor contact ratio: 14.29%
# [35_seg.nii.gz] s51 vein-tumor contact ratio: 2.17%
# [35_seg.nii.gz] s52 vein-tumor contact ratio: 5.88%
# [35_seg.nii.gz] s53 vein-tumor contact ratio: 1.64%
# [35_seg.nii.gz] s54 vein-tumor contact ratio: 1.85%
# [35_seg.nii.gz] s56 vein-tumor contact ratio: 20.00%
# [35_seg.nii.gz] s57 vein-tumor contact ratio: 19.15%
# [35_seg.nii.gz] s58 vein-tumor contact ratio: 16.67%