import cc3d
import numpy as np
from scipy import ndimage
from skimage import morphology

import graph_v7 as g7
import SimpleITK as sitk
import tqdm

from toolkit_3D import get_nii, get_contour_nii, tuple_to_list


def get_target_img(target_only_filename):
    img = sitk.ReadImage(target_only_filename)
    img = sitk.GetArrayFromImage(img)
    # background - 0
    # target - 1
    return img


def get_point_in_list(img, print_info=False):
    contact_point_list = []
    indices = np.where(img == 1)
    if print_info:
        print("contact_points: ")
    for i, j, k in zip(list(indices[0]), list(indices[1]), list(indices[2])):
        contact_point = (i, j, k)
        contact_point_list.append(contact_point)
        if print_info:
            print(contact_point)
    return contact_point_list


def get_contact_related_list(contact_point_list, path_list, img_target, process_name="", print_info=False):
    contact_related_list = []
    for idx, contact_point in enumerate(tqdm.tqdm(contact_point_list, desc=process_name)):
        path_id, path_point_index, _ = classify_to_path(contact_point, path_list, img_target)
        contact_point_dict = {
            "contact_point": contact_point,
            "path_id": path_id,
            "path_point_index": path_point_index
        }
        contact_related_list.append(contact_point_dict)
        if print_info:
            print(str(contact_point) + ": at path " + str(path_id) + ", index = " + str(path_point_index))
    return contact_related_list


def classify_to_path(contact_point, path_list, img_target):
    """
    Classify a contact (surface contour) point, decide which path it belongs to
    :param contact_point:
    :param path_list:
    :param img_target:
    :return:
    """

    def point_distance(point1, point2):
        """
        Calculate the square of distance of two points
        """
        (a1, b1, c1) = point1
        (a2, b2, c2) = point2
        return (a1 - a2) ^ 2 + (b1 - b2) ^ 2 + (c1 - c2) ^ 2

    def point_belong_ratio(point1, point2, img_target):
        """
        Calculate how many points between a contact (surface contour) point and a path point belong to the target,
        and compare the number with the total number of points between, from which we get a result ratio

        Return the [point_list_between] to assist other inner point (in this line) to classify
        :param point1:
        :param point2:
        :param img_target:
        :return: belong_ratio, point_list_between
        """
        point_list_between = get_point_list_between(point1, point2)
        total_point_num = len(point_list_between)
        belong_point_num = 0
        if total_point_num == 0:
            print("Something wrong with [get_point_list_between]")
            exit(1)

        for (x, y, z) in point_list_between:
            if img_target[x, y, z] == 1:
                belong_point_num += 1

        belong_ratio = belong_point_num / total_point_num

        return belong_ratio, point_list_between

    distance = 100000000
    belong_ratio_threshold = 0.9
    path_id = -1
    path_point_index = -1
    same_point_list = None

    for path_dict in path_list:
        path = path_dict["path"]
        for idx, path_point in enumerate(path):

            belong_ratio, point_list_between = point_belong_ratio(contact_point, path_point, img_target)
            if belong_ratio >= belong_ratio_threshold:
                new_distance = point_distance(contact_point, path_point)
                if new_distance < distance:
                    distance = new_distance
                    path_id = path_dict["id"]
                    path_point_index = idx
                    same_point_list = point_list_between
    return path_id, path_point_index, same_point_list


def get_point_list_between(point1, point2):
    """
    Get the points between two given points according to a straight line
    :param point1: Suggest the contact point
    :param point2: Suggest the skeleton point
    :return: Point list between
    """
    (x1, y1, z1) = point1
    (x2, y2, z2) = point2
    x_distance = abs(x1 - x2)
    y_distance = abs(y1 - y2)
    z_distance = abs(z1 - z2)

    # print(str(point1) + ", " + str(point2))
    point_list = []

    if point1 == point2:
        point_list.append(point1)
    elif max(x_distance, y_distance, z_distance) == x_distance:

        distance = x_distance
        ranges = range(0, distance + 1)

        if x1 >= x2:
            x_start = x2
            y_start = y2
            z_start = z2
            k_y = (y1 - y2) / distance
            k_z = (z1 - z2) / distance
        else:
            x_start = x1
            y_start = y1
            z_start = z1
            k_y = (y2 - y1) / distance
            k_z = (z2 - z1) / distance

        for i in ranges:
            x_curr = x_start + i
            y_curr = round(y_start + k_y * i)
            z_curr = round(z_start + k_z * i)
            point_list.append((x_curr, y_curr, z_curr))

    elif max(x_distance, y_distance, z_distance) == y_distance:

        distance = y_distance
        ranges = range(0, distance + 1)

        if y1 >= y2:
            x_start = x2
            y_start = y2
            z_start = z2
            k_x = (x1 - x2) / distance
            k_z = (z1 - z2) / distance
        else:
            x_start = x1
            y_start = y1
            z_start = z1
            k_x = (x2 - x1) / distance
            k_z = (z2 - z1) / distance

        for i in ranges:
            x_curr = round(x_start + k_x * i)
            y_curr = y_start + i
            z_curr = round(z_start + k_z * i)
            point_list.append((x_curr, y_curr, z_curr))

    elif max(x_distance, y_distance, z_distance) == z_distance:

        distance = z_distance
        ranges = range(0, distance + 1)

        if z1 >= z2:
            x_start = x2
            y_start = y2
            z_start = z2
            k_x = (x1 - x2) / distance
            k_y = (y1 - y2) / distance
        else:
            x_start = x1
            y_start = y1
            z_start = z1
            k_x = (x2 - x1) / distance
            k_y = (y2 - y1) / distance

        for i in ranges:
            x_curr = round(x_start + k_x * i)
            y_curr = round(y_start + k_y * i)
            z_curr = z_start + i
            point_list.append((x_curr, y_curr, z_curr))

    else:
        print("Something wrong with [get_point_list_between]")
        return None
    return point_list


def get_list_range(num_list):
    """
    Get the maximum and minimum value of a number list
    :param num_list:
    :return: minimum, maximum
    """
    if len(num_list) == 0:
        print("Something wrong with [get_list_range]")
        exit(1)
    min_num = num_list[0]
    max_num = num_list[0]
    for num in num_list:
        if num < min_num:
            min_num = num
        if num > max_num:
            max_num = num
    return min_num, max_num


def decrypt_index_list(index_list):
    if len(index_list) == 0:
        print("Something wrong with [decrypt_index_list]")
        exit(1)
    idx_list = index_list.copy()
    idx_list = list(set(idx_list))
    idx_list.sort()
    idx_threshold = 4

    start = 0
    new_lists = []
    for i in range(0, len(idx_list)):
        if i == len(idx_list) - 1:
            new_list = idx_list[start: i + 1]
            new_list_dict = {
                "id": i + 1,
                "list": new_list,
                "length": new_list[-1] - new_list[0] + 1,
                "min": new_list[0],
                "max": new_list[-1]
            }
            new_lists.append(new_list_dict)
            break

        if idx_list[i + 1] - idx_list[i] >= idx_threshold:
            new_list = idx_list[start: i + 1]
            start = i + 1
            new_list_dict = {
                "id": 1,
                "list": new_list,
                "length": new_list[-1] - new_list[0] + 1,
                "min": new_list[0],
                "max": new_list[-1]
            }
            new_lists.append(new_list_dict)
    return new_lists


def depict_paths(path_list, img_shape):
    result = np.zeros(img_shape)

    for path_dict in path_list:
        id = path_dict["id"]
        if id != 42:
            continue
        for point in path_dict["path"]:
            result[point] = id
    kernel = morphology.ball(1)
    result = morphology.dilation(result, kernel)
    return result


def skeleton_analysis(file_dict, target, print_info=False):
    origin_img_path = file_dict["img_path"]
    contact_img_path = file_dict["img_contact_path"]

    # get contour dict (contains contour and contact images)
    img_contact_dict = get_contour_nii(contact_img_path)
    img_origin_dict = get_nii(origin_img_path)

    # get target-only one-hot mask image
    img_target = get_nii(origin_img_path)[target]
    img_shape = img_target.shape
    print("image shape: " + str(img_shape))

    # get contact surface point in form of list
    contact_point_list = get_point_in_list(img_contact_dict["contact_" + target], print_info=False)

    # get skeleton information: point_list - cluster points ; path_list - points in paths
    point_list, path_list = get_skeleton_summary(img_origin_dict[target], print_info=False)

    # link contact surface points with skeleton path points
    process_name = "image " + file_dict["img_id"] + " " + target + " processing: "
    contact_related_list = get_contact_related_list(contact_point_list, path_list, img_target,
                                                    process_name=process_name,
                                                    print_info=False)

    result_list = []
    # print(path_list)
    for path_id in range(1, len(path_list) + 1):
        # to get the correspondent point in one path
        p = path_list[path_id - 1]
        index_in_single_path_list = []
        for cpoint in contact_related_list:
            if cpoint["path_id"] == path_id:
                index_in_single_path_list.append(cpoint["path_point_index"])
        if len(index_in_single_path_list) > 0:
            decrypted_list = decrypt_index_list(index_in_single_path_list)
            result_dict = {
                "path_id": path_id,
                "path_length": p["length"],
                "path_start_point": p["start_point"],
                "path_end_point": p["end_point"],
                "path": p["path"],
                "decrypted_list": decrypted_list
            }
            result_list.append(result_dict)
            if print_info:
                print("path_id: " + str(path_id))
                print("    total list: " + str(index_in_single_path_list))
                print(path_list[path_id]["path"])
                print("    is decrypted as: ")
                for d in decrypted_list:
                    print(
                        "        length = " + str(d["length"]) + ", range = [" + str(d["min"]) + ", " + str(
                            d["max"]) + "]")
                    print("            " + str(d["list"]))
    return result_list


def vessel_segment(img, print_info=False):
    # get two kinds of point list
    img_cc3d_distance_transform = ndimage.distance_transform_edt(img)
    contour_img = np.where(img_cc3d_distance_transform == 1, 1, 0)
    inner_img = np.where(img_cc3d_distance_transform > 1, 1, 0)
    contour_point_list = get_point_in_list(contour_img, print_info=print_info)
    inner_point_list = get_point_in_list(inner_img, print_info=print_info)

    # get skeleton information: point_list - cluster points ; path_list - points in paths
    _, path_list = get_skeleton_summary(img, print_info=print_info)

    new_img = np.zeros(img.shape)

    # link points with skeleton path points
    process_name = "Contour Processing: "
    for idx, point in enumerate(tqdm.tqdm(contour_point_list, desc=process_name)):
        if new_img[point] != 0:
            continue
        path_id, path_point_index, same_point_list = classify_to_path(point, path_list, img)
        new_img[point] = path_id
        for same_point in same_point_list:
            if new_img[same_point] != 0:
                new_img[same_point] = path_id

    process_name = "Inner Processing: "
    for idx, point in enumerate(tqdm.tqdm(inner_point_list, desc=process_name)):
        if new_img[point] != 0:
            continue
        path_id, path_point_index, same_point_list = classify_to_path(point, path_list, img)
        new_img[point] = path_id
        for same_point in same_point_list:
            if new_img[same_point] != 0:
                new_img[same_point] = path_id

    return new_img


def get_skeleton_img(img, expand=0):
    """
    Get skeletonized image

    :param img: the image of one target only in numpy
    :param expand: expand the skeleton for visibility, 0 for no-expand
    :return: the image of one skeleton in numpy
    """
    _, path_list = get_skeleton_summary(img, print_info=False)

    path_list = remove_small_skeleton(img, path_list, length_threshold=10)

    skeleton_img = np.zeros(img.shape)
    for path_dict in path_list:
        path_id = path_dict["id"]
        for point in path_dict["path"]:
            skeleton_img[point] = path_id

    if expand > 0:
        kernel = morphology.ball(expand)
        skeleton_img = morphology.dilation(skeleton_img, kernel)
    return skeleton_img


def remove_small_skeleton(img, path_list, length_threshold=4):
    """
    Use cc3d to filter the small skeleton islands, and separate one unconnected skeleton path into two or more

    :param img: the origin one-target image in numpy
    :param path_list: the output path_list of [get_skeleton_summary]
    :param length_threshold: the minium length of one independent skeleton path, default 4
    :return: a new path_list
    """
    new_id = 1
    new_path_list = []
    for path_dict in path_list:
        skeleton_img = np.zeros(img.shape)
        for point in path_dict["path"]:
            skeleton_img[point] = 1

        total_skeleton_length = np.sum(skeleton_img == 1)
        if total_skeleton_length <= length_threshold:
            continue

        skeleton_img_cc3d = cc3d.connected_components(skeleton_img)
        i = 1
        while True:
            total_num = np.sum(skeleton_img_cc3d == i)
            if total_num == 0:
                break
            if total_num > length_threshold:
                new_path = tuple_to_list(np.where(skeleton_img_cc3d == i))
                new_path_dict = {
                    "id": new_id,
                    "length": len(new_path),
                    "start_point": new_path[0],
                    "end_point": new_path[-1],
                    "path": new_path,
                    "weight": -1
                }
                new_id += 1
                new_path_list.append(new_path_dict)
            i += 1
    return new_path_list


def get_path_weight(img, path_list):
    """
    Inject path weight into path_list according to the rough vessel thickness (radis)

    :param img: the target-only image in numpy
    :param path_list: the output path_list of [get_skeleton_summary]
    :return: a new path_list
    """
    new_path_list = []
    shape = img.shape
    large_num = shape[0] ** 2 + shape[1] ** 2 + shape[2] ** 2
    img_neg_mask = np.where(img == 1, 0, 1)

    for path_dict in path_list:
        length = path_dict["length"]
        path = path_dict["path"]
        idx = int((length - 1) / 2)

        root_point = path[idx]
        root_map = np.ones(img.shape)
        root_map[root_point] = 0
        root_map = ndimage.distance_transform_edt(root_map)
        root_map = np.multiply(root_map, img_neg_mask)
        root_map = np.where(root_map == 0, large_num, root_map)
        weight = np.min(root_map)

        new_path_dict = path_dict.copy()
        new_path_dict["weight"] = weight
        new_path_list.append(new_path_dict)
        # print(str(new_path_dict["id"]) + ": " + str(new_path_dict["weight"]))
    return new_path_list


def distance_to_category(path_list, img, weight_threshold=1, weight_decay=1):
    """
    After calculating weight of every path and saving them in the path_list, all the voxels of target organ can be
    sorted according to paths with this function.

    :param path_list: the path_list after [get_path_weight]
    :param img: origin image of single target
    :param weight_decay: how much the weight of paths values to the sort, must within [0, 1], default 1
    :return: category map
    """

    def distance_softmax(path_map_list, origin_img):
        """
        [path_map_list] includes the [path_map]s of every category. For every voxel, this function chooses the minium
        value among all [path_map]s and record the path ID in the value of [category_map] in this voxel position.

        :param path_map_list: path_maps of every category
        :param origin_img: origin image of single target
        :return: a new image, in which target organ is separated according to skeleton
        """
        img_shape = origin_img.shape
        min_distance_init = img_shape[0] ^ 2 + img_shape[1] ^ 2 + img_shape[2] ^ 2
        category_map = np.zeros(img_shape)
        for i in range(0, img_shape[0]):
            for j in range(0, img_shape[1]):
                for k in range(0, img_shape[2]):
                    if origin_img[i, j, k] == 0:
                        continue
                    min_distance = min_distance_init
                    path_id = 0
                    for idx, path_map in path_map_list:
                        if min_distance > path_map[i, j, k]:
                            min_distance = path_map[i, j, k]
                            path_id = idx
                    category_map[i, j, k] = path_id
        return category_map

    path_map_list = []
    for path_dict in path_list:
        if path_dict["weight"] <= weight_threshold:
            continue
        path_map = np.ones(img.shape)
        for point in path_dict["path"]:
            path_map[point] = 0
        path_map = ndimage.distance_transform_edt(path_map) / (path_dict["weight"]) ** weight_decay
        path_map_list.append((path_dict["id"], path_map))

    category_map = distance_softmax(path_map_list, img)

    return category_map


def get_skeleton_summary(img, print_info=False):
    return g7.get_skeleton_summary(img, print_info)


if __name__ == '__main__':

    origin_filename = "data/1_seg.nii.gz"
    # contact_filename = "contact_" + origin_filename
    contact_filename = "data_contact/contact_1_seg.nii.gz"

    target = "artery"

    # get contour dict (contains contour and contact images)
    img_contact_dict = get_contour_nii(contact_filename)
    img_origin_dict = get_nii(origin_filename)

    # get target-only one-hot mask image
    img_target = get_nii(origin_filename)[target]
    img_shape = img_target.shape
    print("image shape: " + str(img_shape))

    # get contact surface point in form of list
    contact_point_list = get_point_in_list(img_contact_dict[target])

    # get skeleton information: point_list - cluster points ; path_list - points in paths
    point_list, path_list = get_skeleton_summary(img_origin_dict[target], print_info=False)

    # link contact surface points with skeleton path points
    contact_related_list = get_contact_related_list(contact_point_list, path_list, img_target, "img 1",
                                                    print_info=False)

    for path_id in range(0, len(path_list)):
        # to get the correspondent point in one path
        index_in_single_path_list = []
        for cpoint in contact_related_list:
            if cpoint["path_id"] == path_id:
                index_in_single_path_list.append(cpoint["path_point_index"])
        if len(index_in_single_path_list) > 0:
            decrypted_list = decrypt_index_list(index_in_single_path_list)
            print("path_id: " + str(path_id))
            print("    total list: " + str(index_in_single_path_list))
            print(path_list[path_id]["path"])
            print("    is decrypted as: ")
            for d in decrypted_list:
                print(
                    "        length = " + str(d["length"]) + ", range = [" + str(d["min"]) + ", " + str(d["max"]) + "]")
                print("            " + str(d["list"]))

    # img_path = depict_paths(path_list, img_shape)
    # new_nii = sitk.GetImageFromArray(img_path.astype(np.uint8))
    # sitk.WriteImage(new_nii, "img_path_42.nii.gz")
