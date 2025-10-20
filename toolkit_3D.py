import logging
import os
from math import sqrt

import cc3d
import numpy as np
import SimpleITK as sitk
from scipy import ndimage
from skimage import morphology
from scipy.spatial.transform import Rotation

def get_nii(filename, axis='z'):
    # read nii 3D data
    img = sitk.ReadImage(filename)

    direction = img.GetDirection()
    spacing = img.GetSpacing()
    origin = img.GetOrigin()
    img_info = (direction, spacing, origin)

    img = sitk.GetArrayFromImage(img)

    if axis == 'z':
        pass
    elif axis == 'x':
        img = img.transpose(1, 0, 2)  # z <-> x
    elif axis == 'y':
        img = img.transpose(2, 1, 0)  # z <-> y

    # separate labels
    img_artery = np.where(img == 1, 1, 0).astype(np.float32)
    img_tumor = np.where(img == 2, 1, 0).astype(np.float32)
    img_vein = np.where(img == 3, 1, 0).astype(np.float32)
    img_pancreas = np.where(img == 4, 1, 0).astype(np.float32)
    img_duct = np.where(img == 5, 1, 0).astype(np.float32)

    result_dict = {
        "origin": img.astype(np.float32),
        "artery": img_artery,
        "tumor": img_tumor,
        "vein": img_vein,
        "pancreas": img_pancreas,
        "duct": img_duct,
        "info": img_info
    }

    return result_dict


def get_contour_nii(contact_filename, axis='z'):
    # background - 0
    # tumor - 1
    # contour_vein - 2
    # contact_vein - 3
    # contour_artery - 4
    # contact_artery - 5
    print(contact_filename)
    img = sitk.ReadImage(contact_filename)
    direction = img.GetDirection()
    spacing = img.GetSpacing()
    origin = img.GetOrigin()
    img_info = (direction, spacing, origin)
    img = sitk.GetArrayFromImage(img)

    if axis == 'z':
        pass
    elif axis == 'x':
        img = img.transpose(1, 0, 2)  # z <-> x
    elif axis == 'y':
        img = img.transpose(2, 1, 0)  # z <-> y

    tumor = np.where(img == 1, 1, 0)
    contour_vein = np.where(img == 2, 1, 0)
    contact_vein = np.where(img == 3, 1, 0)
    contour_artery = np.where(img == 4, 1, 0)
    contact_artery = np.where(img == 5, 1, 0)

    result_dict = {
        "tumor": tumor,
        "contour_vein": contour_vein + contact_vein,
        "contact_vein": contact_vein,
        "contour_artery": contour_artery + contact_artery,
        "contact_artery": contact_artery,
        "info": img_info
    }

    return result_dict


def get_any_nii(filename, axis='z'):
    img = sitk.ReadImage(filename)
    direction = img.GetDirection()
    spacing = img.GetSpacing()
    origin = img.GetOrigin()
    img_info = (direction, spacing, origin)
    img = sitk.GetArrayFromImage(img)

    if axis == 'z':
        pass
    elif axis == 'x':
        img = img.transpose(1, 0, 2)  # z <-> x
    elif axis == 'y':
        img = img.transpose(2, 1, 0)  # z <-> y

    result_dict = {
        "img": img,
        "info": img_info
    }

    return result_dict


def save_nii(new_img, new_filename, img_info, axis='z'):
    if axis == 'z':
        pass
    elif axis == 'x':
        new_img = new_img.transpose(1, 0, 2)  # x <-> z
    elif axis == 'y':
        new_img = new_img.transpose(2, 1, 0)  # x <-> y

    direction, spacing, origin = img_info
    new_nii = sitk.GetImageFromArray(new_img.astype(np.uint8))
    new_nii.SetSpacing(spacing)
    new_nii.SetOrigin(origin)
    new_nii.SetDirection(direction)
    sitk.WriteImage(new_nii, new_filename)


def get_3D_contour(img, contour_thickness=1.5):
    new_img = ndimage.distance_transform_edt(img)
    new_img = np.where(new_img == 0, contour_thickness + 1, new_img)
    new_img = np.where(new_img < contour_thickness, 1, 0)
    return new_img


def get_2D_contour(img, contour_thickness=1.5):
    deep = img.shape[0]
    new_img = np.zeros(img.shape)
    for slice_num in range(0, deep):
        slice = img[slice_num, :, :]
        new_slice = ndimage.distance_transform_edt(slice)
        new_slice = np.where(new_slice == 0, contour_thickness + 1, new_slice)
        new_slice = np.where(new_slice < contour_thickness, 1, 0)
        new_img[slice_num, :, :] = new_slice
    return new_img


def get_3D_contact(img_tumor, contour_vessel, contact_range=3):
    contact = np.where(img_tumor == 1, 0, 1)
    contact = ndimage.distance_transform_edt(contact)
    contact = np.multiply(contact, contour_vessel)
    contact = np.where(contact > contact_range, 0, contact)
    contact = np.where(contact > 0, 1, 0)
    return contact


def get_2D_contact(img_tumor, contour_vessel, contact_range=1.5):
    deep = img_tumor.shape[0]
    contact = np.zeros(img_tumor.shape)

    for slice_num in range(0, deep):
        slice_contact = np.where(img_tumor == 1, 0, 1)
        slice_contact = ndimage.distance_transform_edt(slice_contact)
        slice_contact = np.multiply(slice_contact, contour_vessel)
        slice_contact = np.where(slice_contact > contact_range, 0, slice_contact)
        slice_contact = np.where(slice_contact > 0, 1, 0)
        contact[slice_num, :, :] = slice_contact
    return contact


def remove_islands(img, threshold_size=1000):
    """
    Remove small islands in the image. The removed islands have less voxel than threshold_size
    If threshold_size < 0, keep only the largest connected island
    :param img:
    :param threshold_size:
    :return:
    """
    new_img = cc3d.connected_components(img)
    if threshold_size >= 0:
        i = 1
        while True:
            total_num = np.sum(new_img == i)
            if total_num == 0:
                break
            if total_num <= threshold_size:
                new_img = np.where(new_img == i, 0, new_img)
            i += 1
        result_img = np.where(new_img > 0, 1, 0)
        return result_img
    else:
        i = 1
        max_total_sum = 0
        max_island = 0
        while True:
            total_num = np.sum(new_img == i)
            if total_num == 0:
                break
            if total_num > max_total_sum:
                max_island = i
                max_total_sum = total_num
            i += 1
        if max_island > 0:
            result_img = np.where(new_img == max_island, 1, 0)
            return result_img
        else:
            print("No target in [remove_islands]")
            exit(1)


def get_islands_num(img):
    new_img = cc3d.connected_components(img)
    return np.max(new_img), new_img


def get_island_info(img):
    isl_num, isl_img = get_islands_num(img)

    isl_size_list = []

    for i in range(1, isl_num + 1):
        isl_size = np.sum(isl_img[isl_img == i])
        isl_size_list.append(isl_size)
    isl_size_list.sort(reverse=True)

    total_size = np.sum(isl_img[isl_img > 0])

    return isl_num, isl_size_list, isl_img, total_size


def tuple_to_list(where_tuple):
    """
    When we use [numpy.where(condition)], ignoring the following 2 parameters, we receive a tuple of list,
    which denotes the coordinate of the target points. But coordinate in this form cannot be used.
    This function translate it into list of tuple like [(0, 1, 0), (1, 0, 1), ...], and currently can translate
    2D and 3D environment

    :param where_tuple: the output of [numpy.where(condition)]
    :return: coordinate in list of tuple
    """
    dim = len(where_tuple)
    point_list = []
    if dim == 2:
        for i, j in zip(where_tuple[0], where_tuple[1]):
            point_list.append((i, j))
    elif dim == 3:
        for i, j, k in zip(where_tuple[0], where_tuple[1], where_tuple[2]):
            point_list.append((i, j, k))
    else:
        print("Error in [tuple_to_list]")
        exit(1)
    return point_list


# def erode(img, times=1):
#     distance_map = ndimage.distance_transform_edt(img)
#     eroded = np.zeros(img.shape)
#     for i in range(0, times):
#         eroded += np.where(distance_map == 1, 1, 0)
#         distance_map = np.where(distance_map <= 1, 0, 1)
#     return distance_map, eroded

def erode(img, times=1):
    distance_map = img.copy()
    eroded = np.zeros(img.shape)
    for i in range(0, times):
        distance_map = ndimage.distance_transform_edt(distance_map)
        eroded += np.where(distance_map == 1, 1, 0)
        distance_map = np.where(distance_map <= 1, 0, 1)
    return distance_map, eroded


def is_connect(img):
    img_cc3d = cc3d.connected_components(img)
    if np.max(img_cc3d) > 1:
        return False
    else:
        return True


def thin_detect(part_img, thin_threshold=1):
    part_id_range = range(1, int(np.max(part_img)) + 1)
    thin_detect_img = np.zeros(part_img.shape)
    for part_id in part_id_range:
        part = np.where(part_img == part_id, 1, 0)
        for thin_times in range(1, thin_threshold + 1):
            part_erode = erode(part, thin_times)
            if not is_connect(part_erode):
                thin_detect_img += part * part_id
                break
    return thin_detect_img


def fill_hole(img):
    new_img = np.where(img == 1, 0, 1)
    new_img = remove_islands(new_img, threshold_size=-1)
    new_img = np.where(new_img == 1, 0, 1)
    return new_img


def get_sphere_mask(center_point, shape, radis):
    mask = np.ones(shape)
    mask[center_point] = 0
    mask = ndimage.distance_transform_edt(mask)
    mask = np.where(mask <= radis, 1, 0)
    return mask


def get_ellipticity(map):
    shape = map.shape

    range_length = shape[0] + shape[1] - 2
    range_list_f = list(range(0, range_length))
    range_list_b = range_list_f.copy()
    range_list_b.reverse()

    index_min = 0
    index_max = 0

    for i in range_list_f:
        has_land = False
        for j in range(0, i):
            point = (j, i - j)

            if point[0] < 0 or point[0] >= shape[0] or point[1] < 0 or point[1] >= shape[1]:
                continue

            if map[point] > 0:
                has_land = True
                break

        if has_land:
            index_min = i
            break

    for i in range_list_b:
        has_land = False
        for j in range(0, i):
            point = (j, i - j)

            if point[0] < 0 or point[0] >= shape[0] or point[1] < 0 or point[1] >= shape[1]:
                continue

            if map[point] > 0:
                has_land = True
                break

        if has_land:
            index_max = i
            break

    width_lu2rd = (index_max - index_min + 1) * 1.414

    for i in range_list_f:
        has_land = False
        for j in range(0, i):
            point = (shape[0] - 1 - j, i - j)

            if point[0] < 0 or point[0] >= shape[0] or point[1] < 0 or point[1] >= shape[1]:
                continue

            if map[point] > 0:
                has_land = True
                break

        if has_land:
            index_min = i
            break

    for i in range_list_b:
        has_land = False
        for j in range(0, i):
            point = (shape[0] - 1 - j, i - j)

            if point[0] < 0 or point[0] >= shape[0] or point[1] < 0 or point[1] >= shape[1]:
                continue

            if map[point] > 0:
                has_land = True
                break

        if has_land:
            index_max = i
            break

    width_ld2ru = (index_max - index_min + 1) * 1.414

    range_length = shape[0]
    range_list_f = list(range(0, range_length))
    range_list_b = range_list_f.copy()
    range_list_b.reverse()

    for i in range_list_f:
        has_land = False
        for j in range(0, shape[1]):
            point = (i, j)
            if map[point] > 0:
                has_land = True
                break
        if has_land:
            index_min = i
            break

    for i in range_list_b:
        has_land = False
        for j in range(0, shape[1]):
            point = (i, j)
            if map[point] > 0:
                has_land = True
                break
        if has_land:
            index_max = i
            break

    width_u2d = index_max - index_min + 1

    range_length = shape[1]
    range_list_f = list(range(0, range_length))
    range_list_b = range_list_f.copy()
    range_list_b.reverse()

    for i in range_list_f:
        has_land = False
        for j in range(0, shape[0]):
            point = (j, i)
            if map[point] > 0:
                has_land = True
                break
        if has_land:
            index_min = i
            break

    for i in range_list_b:
        has_land = False
        for j in range(0, shape[0]):
            point = (j, i)
            if map[point] > 0:
                has_land = True
                break
        if has_land:
            index_max = i
            break

    width_l2r = index_max - index_min + 1

    width_max = max(width_lu2rd, width_ld2ru, width_u2d, width_l2r)
    width_min = min(width_lu2rd, width_ld2ru, width_u2d, width_l2r)
    width_avg = (width_lu2rd + width_ld2ru + width_u2d + width_l2r) / 4
    ellipticity = (width_max - width_min) / width_avg

    return ellipticity


def get_surround_voxel(coordinate, shape):
    max_cood = (shape[0] - 1, shape[1] - 1, shape[2] - 1)
    relative_list = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (-1, 0, 0), (0, -1, 0), (0, 0, -1),  # 6 faces
                     (1, 1, 0), (1, 0, 1), (1, -1, 0), (1, 0, -1),
                     (-1, 1, 0), (-1, 0, 1), (-1, -1, 0), (-1, 0, -1),
                     (0, 1, 1), (0, 1, -1), (0, -1, 1), (0, -1, -1),  # 12 vertices
                     (1, 1, 1), (1, 1, -1), (1, -1, 1), (-1, 1, 1),
                     (1, -1, -1), (-1, 1, -1), (-1, -1, 1), (-1, -1, -1)  # 8 angels
                     ]
    sur_cood_list = []
    for rel_cood in relative_list:
        sur_cood = (rel_cood[0] + coordinate[0], rel_cood[1] + coordinate[1], rel_cood[2] + coordinate[2])
        sur_cood = (min(sur_cood[0], max_cood[0]), min(sur_cood[1], max_cood[1]), min(sur_cood[2], max_cood[2]))
        sur_cood = (max(sur_cood[0], 0), max(sur_cood[1], 0), max(sur_cood[2], 0))
        sur_cood_list.append(sur_cood)
    sur_cood_list = list(set(sur_cood_list))  # remove same coordinate
    if coordinate in sur_cood_list:
        sur_cood_list.remove(coordinate)
    return sur_cood_list


def is_in_list(element, element_list):
    for e in element_list:
        if e == element:
            return True
    return False


def get_distance(coordinate1, coordinate2, spacing=(1.0, 1.0, 1.0)):
    return sqrt(
        ((coordinate1[0] - coordinate2[0]) * spacing[0]) ** 2
        + ((coordinate1[1] - coordinate2[1]) * spacing[1]) ** 2
        + ((coordinate1[2] - coordinate2[2]) * spacing[2]) ** 2
    )


def get_angle(direction_1, direction_2):
    """
    direction_1 and direction_2 are two 3D unit vector (length = 1), calculate the angle (<90) between
    :param direction_1:
    :param direction_2:
    :return:
    """
    x_1, y_1, z_1 = direction_1
    x_2, y_2, z_2 = direction_2
    cos = x_1 * x_2 + y_1 * y_2 + z_1 * z_2
    angel = np.arccos(cos)
    angel = np.degrees(angel)
    if angel > 90:
        angel = 180 - angel
    return angel


def get_inter_distance(mask1, mask2):
    """
    Giving two 3D mask, calculate the min distance between them
    :param mask1:
    :param mask2:
    :return:
    """
    dist_map = ndimage.distance_transform_edt(np.where(mask1 > 0, 0, 1))
    dist_map = np.multiply(np.where(mask2 > 0, 1, 0), dist_map)
    return np.min(np.where(dist_map == 0, 114514, dist_map))


def image_dilation(image, radius):
    kernel = morphology.ball(radius)
    img_dilation = morphology.dilation(image, kernel)
    return img_dilation


def image_erosion(image, radius):
    kernel = morphology.ball(radius)
    img_erosion = morphology.erosion(image, kernel)
    return img_erosion


def unravel_coordinate(vox_str: str) -> tuple:
    if len(vox_str) <= 4:
        logging.error("Bad string in toolkit_3D.unravel_coordinate()")
        exit(-1)
    vox_str = vox_str.split('(')[1].split(')')[0]
    axis_list = vox_str.split(',')
    coordinate = None
    for axis in axis_list:
        axis = int(axis.strip())
        if coordinate is None:
            coordinate = (axis,)
        else:
            coordinate = coordinate + (axis,)
    return coordinate


def image_rotation(img, center, direction, rotation_range=10):
    # 定义旋转角度和轴
    # vox1 = (53, 121, 81)
    # vox2 = (47, 126, 101)
    # vector = (vox1[0] - vox2[0], vox1[1] - vox2[1], vox1[2] - vox2[2])
    shape = img.shape

    # rotation_axis = [0, 0, 0]
    # if direction[0] == 0:
    #     rotation_axis[0] = 1
    # else:
    #     rotation_axis[0] = np.arctan((direction[1] ** 2 + direction[2] ** 2) ** 0.5 / direction[0])
    # if direction[2] == 0:
    #     rotation_axis[2] = 1
    # else:
    #     rotation_axis[2] = np.arctan((direction[1] ** 2 + direction[0] ** 2) ** 0.5 / direction[2])
    direction_t = (1, 0, 0)
    rotation_axis = (direction[1] * direction_t[2] - direction[2] * direction_t[1],
                     direction[2] * direction_t[0] - direction[0] * direction_t[2],
                     direction[0] * direction_t[1] - direction[1] * direction_t[0])  # Vector cross product
    # rotation_axis = (-rotation_axis[0], -rotation_axis[1], -rotation_axis[2])
    rotation_angle_sin = ((rotation_axis[0] ** 2 + rotation_axis[1] ** 2 + rotation_axis[2] ** 2) /
                          (direction[0] ** 2 + direction[1] ** 2 + direction[2] ** 2)) ** 0.5
    rotation_angle = np.arcsin(rotation_angle_sin)
    # rotation_angle = 3.14 / 2
    print(np.degrees(rotation_angle))
    rotation_axis = np.array((rotation_axis[0] * rotation_angle, rotation_axis[1] * rotation_angle, rotation_axis[2] * rotation_angle))
    # angle_degrees = 45  # 旋转角度（以度为单位）
    # rotation_axis = np.array([1, 0, 0])  # 旋转轴（例如，绕x轴旋转）

    # 定义旋转中心坐标
    # center = np.array([50, 124, 86])  # 例如，这里选择一个点

    # 创建旋转矩阵
    # rotation = R.from_rotvec(np.radians(angle_degrees) * rotation_axis)
    rotation = Rotation.from_rotvec(rotation_axis)

    # 创建一个新的3D数组来存储旋转后的数据
    rotated_img = np.zeros_like(img)

    if rotation_range <= 0:
        range_x = range(0, shape[0])
        range_y = range(0, shape[1])
        range_z = range(0, shape[2])
    else:
        range_x = range(max(center[0] - rotation_range, 0), min(center[0] + rotation_range, shape[0]))
        range_y = range(max(center[1] - rotation_range, 0), min(center[1] + rotation_range, shape[1]))
        range_z = range(max(center[2] - rotation_range, 0), min(center[2] + rotation_range, shape[2]))
    for x in range_x:
        for y in range_y:
            for z in range_z:
                # 计算点相对于旋转中心的偏移
                offset = np.array([x, y, z]) - center

                # 使用旋转矩阵旋转偏移
                rotated_offset = rotation.apply(offset)

                # 计算旋转后的坐标
                rotated_x, rotated_y, rotated_z = rotated_offset + center

                # 检查旋转后的坐标是否在矩阵范围内
                if (0 <= rotated_x < shape[0]
                    and 0 <= rotated_y < shape[1]
                    and 0 <= rotated_z < shape[2]):
                    # 使用插值方法从原始数据中获取旋转后的值
                    rotated_img[x, y, z] = img[int(rotated_x), int(rotated_y), int(rotated_z)]
    return rotated_img


class Plane3D:
    """
    生成一个过三个点的空间平面
    """
    def __init__(self, vox1, vox2, vox3):
        point1 = np.array(list(vox1))
        point2 = np.array(list(vox2))
        point3 = np.array(list(vox3))

        # 两个边的向量
        vector1 = point2 - point1
        vector2 = point3 - point1

        # 平面的法向量
        self.normal_vector = np.cross(vector1, vector2)

        # 平面的偏移参数
        self.D = -np.dot(self.normal_vector, point1)

        # posi_encoding = normal_vector[0] * x + normal_vector[1] * y + normal_vector[2] * z + D

    def get_posi_encoding(self, vox):
        return self.normal_vector[0] * vox[0] + self.normal_vector[1] * vox[1] + self.normal_vector[2] * vox[2] + self.D


def print_color(string, color, end=None):
    color_dict = {
        'red': '\033[91m',
        'green': '\033[92m',
        'yellow': '\033[93m',
        'blue': '\033[94m',
        'purple': '\033[95m',
        'clan': '\033[96m',
    }
    color_end = '\033[0m'
    assert color in color_dict.keys(), 'Wrong color'
    if end is not None:
        print(color_dict[color] + string + color_end, end=end)
    else:
        print(color_dict[color] + string + color_end)


def basic_data_analysis(data_list):
    min_ = np.min(np.array(data_list))
    max_ = np.max(np.array(data_list))
    mean = np.mean(np.array(data_list))
    ptp = np.ptp(np.array(data_list))
    var = np.var(np.array(data_list))
    std = np.std(np.array(data_list))
    return min_, max_, mean, ptp, var, std
