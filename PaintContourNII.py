import os

import nibabel as nib
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import sys
import SimpleITK as sitk

sys.setrecursionlimit(500000)  # 例如这里设置为十万


def get_nii(filename):
    # read nii 3D data
    nii = nib.load(filename)
    img = nii.get_fdata().astype(np.float32)

    # rotate 90 to fit itk-snap
    img = np.rot90(img, k=1, axes=(0, 1))

    # seperate labels
    img_background = np.where(img == 0, 1, 0).astype(np.float32)
    img_artery = np.where(img == 1, 1, 0).astype(np.float32)
    img_tumor = np.where(img == 2, 1, 0).astype(np.float32)
    img_vein = np.where(img == 3, 1, 0).astype(np.float32)

    affine = nii.affine.copy()
    hdr = nii.header.copy()

    result_dict = {
        "origin": img.astype(np.float32),
        "background": img_background,
        "artery": img_artery,
        "tumor": img_tumor,
        "vein": img_vein,
        "affine": affine,
        "header": hdr
    }

    return result_dict


def get_nii_slices(img0, axis="x"):

    result_list = []

    if axis == "x":
        max_range = img0.shape[0]
    elif axis == "z":
        max_range = img0.shape[2]
    else:
        return None

    for i in range(0, max_range):
        # get 2D slices
        if axis == "x":
            img = img0[i, :, :]
        elif axis == "z":
            img = img0[:, :, i]
        else:
            return None

        # separate labels
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

        result_list.append(result_dict)

    return result_list


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


def contour_detect_3D(img0, target, print_info=False):
    z_list = get_nii_slices(img0, axis="z")
    x_list = get_nii_slices(img0, axis="x")

    rebuild_z = None

    for i, z_dict in enumerate(z_list):

        if rebuild_z is None:
            rebuild_z = contours_detect(z_dict[target])
            rebuild_z = np.expand_dims(rebuild_z, axis=2)
        else:
            new_slice = contours_detect(z_dict[target])
            new_slice = np.expand_dims(new_slice, axis=2)
            rebuild_z = np.concatenate([rebuild_z, new_slice], axis=2)

    if print_info:
        print("Contour rebuilding on z [" + filename + "] size: " + str(rebuild_z.shape))

    rebuild_x = None

    for i, x_dict in enumerate(x_list):

        if rebuild_x is None:
            rebuild_x = contours_detect(x_dict[target])
            rebuild_x = np.expand_dims(rebuild_x, axis=0)
        else:
            new_slice = contours_detect(x_dict[target])
            new_slice = np.expand_dims(new_slice, axis=0)
            rebuild_x = np.concatenate([rebuild_x, new_slice], axis=0)

    if print_info:
        print("Contour rebuilding on x [" + filename + "] size: " + str(rebuild_x.shape))

    contour = np.logical_or(rebuild_x, rebuild_z).astype(np.float32)

    return contour


def expand_3D(array, level=1):
    expand_img = np.array(array, copy=True)
    indices = np.where(expand_img == 1)
    if level == 1:
        expand_list = [(-1, 0, 0), (1, 0, 0),
                       (0, -1, 0), (0, 1, 0),
                       (0, 0, -1), (0, 0, 1)]
    elif level == 2:
        expand_list = []
        for i in range(-2, 3):
            for j in range(-2, 3):
                for k in range(-2, 3):
                    if i ^ 2 + j ^ 2 + k ^ 2 > 6:
                        continue
                    else:
                        expand_list.append((i, j, k))

        # expand_list = [(-1, 1, -2), (-1, 0, -2),
        #
        #                (-2, 1, -1), (-2, 0, -1), (-2, -1, -1),
        #                (-1, 2, -1), (-1, 1, -1), (-1, 0, -1), (-1, -1, -1), (-1, -2, -1),
        #                (0, 2, -1), (0, 1, -1), (0, -1, -1), (0, -2, -1),
        #                (1, 2, -1), (1, 1, -1), (1, 0, -1), (1, -1, -1), (1, -2, -1),
        #                (2, 1, -1), (2, 0, -1), (2, -1, -1),
        #
        #                (-2, 1, 0), (-2, 0, 0), (-2, -1, 0),
        #                (-1, 2, 0), (-1, 1, 0), (-1, 0, 0), (-1, -1, 0), (-1, -2, 0),
        #                (0, 2, 0), (0, 1, 0), (0, -1, 0), (0, -2, 0),
        #                (1, 2, 0), (1, 1, 0), (1, 0, 0), (1, -1, 0), (1, -2, 0),
        #                (2, 1, 0), (2, 0, 0), (2, -1, 0),
        #
        #                (-2, 1, 1), (-2, 0, 1), (-2, -1, 1),
        #                (-1, 2, 1), (-1, 1, 1), (-1, 0, 1), (-1, -1, 1), (-1, -2, 1),
        #                (0, 2, 1), (0, 1, 1), (0, -1, 1), (0, -2, 1),
        #                (1, 2, 1), (1, 1, 1), (1, 0, 1), (1, -1, 1), (1, -2, 1),
        #                (2, 1, 1), (2, 0, 1), (2, -1, 1),
        #                ]

        # [ ][X][X][X][ ]
        # [X][X][X][X][X]
        # [X][X][O][X][X]
        # [X][X][X][X][X]
        # [ ][X][X][X][ ]

        # [ ][X][X][X][ ]
        # [X][X][X][X][X]
        # [X][X][O][X][X]
        # [x][X][X][X][X]
        # [ ][X][X][X][ ]

        # [ ][ ][ ][ ][ ]
        # [ ][X][X][X][ ]
        # [ ][X][O][X][ ]
        # [ ][X][X][X][ ]
        # [ ][ ][ ][ ][ ]
    else:
        return

    for i, j, k in zip(list(indices[0]), list(indices[1]), list(indices[2])):
        for (a, b, c) in expand_list:
            if i + a > array.shape[0] - 1 or i + a < 0 \
                    or j + b > array.shape[1] - 1 or j + b < 0 \
                    or k + c > array.shape[0] - 1 or k + c < 0:
                continue
            expand_img[i + a][j + b][k + c] = 1
    return expand_img


def get_level_list(level):
    level_list = []
    for i in range(- level, level + 1):
        for j in range(- level, level):
            level_list.append((i, j))
    return level_list


def sum_surrounding(array, x, y):
    array_height = array.shape[1]
    array_width = array.shape[0]

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


def get_contact_related_contours(contour, contact, z_range, ignore_empty=False):
    def according_to_adjacent_z(result, contour, z, position):
        if position == "top":
            detect_z = z + 1
        elif position == "bottom":
            detect_z = z - 1
        else:
            return None
        for i in range(0, contour.shape[0]):
            for j in range(0, contour.shape[1]):
                if contour[i, j, z] == 1 and result[i, j, detect_z] \
                        + result[i, j + 1, detect_z] + result[i, j - 1, detect_z] \
                        + result[i + 1, j, detect_z] + result[i - 1, j, detect_z] \
                        + result[i + 1, j + 1, detect_z] + result[i + 1, j - 1, detect_z] \
                        + result[i - 1, j + 1, detect_z] + result[i - 1, j - 1, detect_z] > 0:
                    return i, j
        return None

    def mark(contour, result, x, y, z):
        if contour[x, y, z] == result[x, y, z]:
            return
        else:
            result[x, y, z] = 1
            xy_iter(contour, result, x, y, z)

    def xy_iter(contour, result, x, y, z):
        if x > 0:
            mark(contour, result, x - 1, y, z)
        if y > 0:
            mark(contour, result, x, y - 1, z)
        if x < contour.shape[1] - 1:
            mark(contour, result, x + 1, y, z)
        if y < contour.shape[0] - 1:
            mark(contour, result, x, y + 1, z)
        if x > 0 and y > 0:
            mark(contour, result, x - 1, y - 1, z)
        if x < contour.shape[1] - 1 and y < contour.shape[0] - 1:
            mark(contour, result, x + 1, y + 1, z)
        if x > 0 and y < contour.shape[0] - 1:
            mark(contour, result, x + 1, y - 1, z)
        if y > 0 and x < contour.shape[1] - 1:
            mark(contour, result, x - 1, y + 1, z)

    def get_contact_related_contours_2D(contour, contact):
        result = np.zeros(contact.shape)

        indices = np.where(contact == 1)

        for i, j, k in zip(list(indices[0]), list(indices[1]), list(indices[2])):
            mark(contour, result, i, j, k)
        return result

    z_min = z_range[0]
    z_max = z_range[1]

    print("Concentration range on axis-z: [" + str(z_min) + ", " + str(z_max) + "]")
    print()

    result = get_contact_related_contours_2D(contour, contact)

    empty_list = []
    for zi in range(z_min, z_max + 1):
        z_layer = result[:, :, zi]
        if np.sum(z_layer[z_layer == 1]) == 0:
            empty_list.append(zi)

    print("These layers z do not contain any [contact] pixel:")
    print("z = " + str(empty_list))
    print()

    if ignore_empty:
        return result

    empty_list_1 = []
    empty_list_2 = []

    for zi in empty_list:
        a = according_to_adjacent_z(result, contour, zi, position="top")
        if a is None:
            empty_list_1.append(zi)
            continue
        x, y = a
        mark(contour, result, x, y, zi)

    for zi in empty_list_1:
        a = according_to_adjacent_z(result, contour, zi, position="bottom")
        if a is None:
            empty_list_2.append(zi)
            continue
        x, y = a
        mark(contour, result, x, y, zi)

    print("These layers z do not contain any [related target] pixel:")
    print("z = " + str(empty_list_2))
    print()

    return result


def show_result(img, contact, contours, filename, sliceNum):
    result_img = img.copy()
    result_img += contours * 20
    result_img -= contact * 7
    ax = plt.matshow(result_img)

    # 计算接触/周长比例
    total_contact_contours = np.sum(contact[contact == 1])
    total_contours = np.sum(contours[contours == 1])
    contact_ratio = total_contact_contours / total_contours

    plt.colorbar(ax.colorbar, fraction=0.025)
    plt.title("z-slice {}".format(sliceNum) + " in " + filename + "\n" + "vein-tumor contact ratio: {:.2%}".format(
        contact_ratio))
    plt.show()

    return contact_ratio


def show_3D_with_slider(img, max_height=600, max_width=800):
    def img_intensity_change_x(x):
        pass

    img_height = img.shape[0]
    img_width = img.shape[1]
    img_depth = img.shape[2]

    ratio = 1

    actual_width = max_width
    actual_height = max_height

    if img_width > img_height:
        ratio = max_width / img_width
        actual_height = img_height * ratio
    else:
        ratio = max_height / img_height
        actual_width = max_width * ratio

    actual_width = int(actual_width)
    actual_height = int(actual_height)

    slice_num = 0

    cv.namedWindow('img', cv.WINDOW_NORMAL)
    cv.resizeWindow("img", actual_width, actual_height)

    # 第一个参数是滑动杆名称，第二个是对应的图片，第三个是默认值，第四个是最大值，第五个是回调函数
    cv.createTrackbar('slice', 'img', slice_num, img_depth - 1, img_intensity_change_x)

    while 1:

        # 拿到对应滑动杆的值
        slice_num = cv.getTrackbarPos('slice', 'img')

        img_slice = img[:, :, slice_num]

        cv.imshow('img', img_slice)
        # 每1毫秒刷新一次，当输入q键的时候，结束整个主程序
        if cv.waitKey(1) == ord('q'):
            break

    cv.destroyAllWindows()


def save_nii(source_filename, destnation, new_data):
    nii_img = nib.load(source_filename)
    new_data = np.rot90(new_data, k=-1, axes=(0, 1))
    affine = nii_img.affine.copy()
    hdr = nii_img.header.copy()
    new_nii = nib.Nifti1Image(new_data, affine, hdr)

    nib.save(new_nii, destnation)


def generate_contour_nii(file_dict, destnation_path=None):
    """
    save as [contact_filename]
    background - 0
    tumor - 1
    contour_vein - 2
    contact_vein - 3
    contour_artery - 4
    contact_artery - 5
    :param img_path:
    :return:
    """
    img_path = file_dict["img_path"]
    if destnation_path is None:
        destnation_path = os.path.split(img_path)[0]
    filename = os.path.split(img_path)[1]

    # read nii image
    img_dict = get_nii(img_path)

    # get target contour
    contour_vein = contour_detect_3D(img_dict["origin"], "vein", print_info=False)
    contour_artery = contour_detect_3D(img_dict["origin"], "artery", print_info=False)

    # calculate target-tumor contact
    tumor = img_dict["tumor"]
    tumor_expand = expand_3D(tumor, 1)
    contact_vein = np.logical_and(tumor_expand, contour_vein).astype(np.float32)
    contact_artery = np.logical_and(tumor_expand, contour_artery).astype(np.float32)

    new_data = tumor * 1 + contour_vein * 2 + contact_vein + contour_artery * 4 + contact_artery

    destnation = os.path.join(destnation_path, "contact_" + str(filename))
    file_dict["img_contact_path"] = destnation

    save_nii(img_path, destnation, new_data)


if __name__ == '__main__':

    # filename = "35_seg.nii.gz"
    # filename = "32_seg.nii.gz"
    filename = "30_seg.nii.gz"
    # filename = "2_seg.nii.gz"
    # filename = "3_seg.nii.gz"

    # read nii image
    img_dict = get_nii(filename)

    # get target contour
    contour_vein = contour_detect_3D(img_dict["origin"], "vein", print_info=False)
    contour_artery = contour_detect_3D(img_dict["origin"], "artery", print_info=False)

    # calculate target-tumor contact
    tumor = img_dict["tumor"]
    tumor_expand = expand_3D(tumor, 1)
    contact_vein = np.logical_and(tumor_expand, contour_vein).astype(np.float32)
    contact_artery = np.logical_and(tumor_expand, contour_artery).astype(np.float32)

    new_data = tumor * 1 + contour_vein * 2 + contact_vein + contour_artery * 4 + contact_artery

    # background - 0
    # tumor - 1
    # contour_vein - 2
    # contact_vein - 3
    # contour_artery - 4
    # contact_artery - 5

    save_nii(filename, new_data)











"""
3D 处理想法：
1. 在 3D 环境下处理，直接分离出分立的血管，按血管保留计算接触
2. 研究 3D 环境下的边界处理和交界面处理
3. 在一定的轴向区域中考虑比例问题

"""
