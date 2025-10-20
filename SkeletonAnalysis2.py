import sys

import numpy as np

from graph_v7 import get_skeleton_summary
import SimpleITK as sitk
sys.setrecursionlimit(500000)  # 例如这里设置为十万


def get_contour_img(contact_filename):
    img = sitk.ReadImage(contact_filename)
    img = sitk.GetArrayFromImage(img)
    # background - 0
    # tumor - 1
    # contour_vein - 2
    # contact_vein - 3
    # contour_artery - 4
    # contact_artery - 5

    tumor = np.where(img == 1, 1, 0)
    contour_vein = np.where(img == 2, 1, 0)
    contact_vein = np.where(img == 3, 1, 0)
    contour_artery = np.where(img == 4, 1, 0)
    contact_artery = np.where(img == 5, 1, 0)

    result_dict = {
        "tumor": tumor,
        "contour_vein": contour_vein,
        "contact_vein": contact_vein,
        "contour_artery": contour_artery,
        "contact_artery": contact_artery
    }

    return result_dict


def get_target_img(target_only_filename):
    img = sitk.ReadImage(target_only_filename)
    img = sitk.GetArrayFromImage(img)
    # background - 0
    # target - 1
    return img


def get_contact_point_list(img_dict, target, print_info=False):
    contact_point_list = []
    indices = np.where(img_dict["contact_" + target] == 1)
    if print_info:
        print("contact_points: ")
    for i, j, k in zip(list(indices[0]), list(indices[1]), list(indices[2])):
        contact_point = (i, j, k)
        contact_point_list.append(contact_point)
        if print_info:
            print(contact_point)
    return contact_point_list


def get_contact_related_list(contact_point_list, path_list, img_target, print_info=False):
    contact_related_list = []
    for contact_point in contact_point_list:
        path_id, path_point_index = classify_to_path(contact_point, path_list, img_target)
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
        :param point1:
        :param point2:
        :return:
        """
        (a1, b1, c1) = point1
        (a2, b2, c2) = point2
        return (a1 - a2) ^ 2 + (b1 - b2) ^ 2 + (c1 - c2) ^ 2

    def point_belong_ratio(point1, point2, img_target):
        """
        Calculate how many points between a contact (surface contour) point and a path point belong to the target,
        and compare the number with the total number of points between, from which we get a result ratio
        :param point1:
        :param point2:
        :param img_target:
        :return: belong_ratio
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

        return belong_ratio

    distance = 100000000
    belong_ratio_threshold = 0.9
    path_id = -1
    path_point_index = -1

    for path_dict in path_list:
        path = path_dict["path"]
        for idx, path_point in enumerate(path):

            belong_ratio = point_belong_ratio(contact_point, path_point, img_target)
            if belong_ratio >= belong_ratio_threshold:
                new_distance = point_distance(contact_point, path_point)
                if new_distance < distance:
                    distance = new_distance
                    path_id = path_dict["id"]
                    path_point_index = idx
    return path_id, path_point_index


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

    point_list = []

    if max(x_distance, y_distance, z_distance) == x_distance:

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
            new_list = idx_list[start : i + 1]
            new_list_dict = {
                "list": new_list,
                "length": len(new_list),
                "min": new_list[0],
                "max": new_list[-1]
            }
            new_lists.append(new_list_dict)
            break
        if idx_list[i + 1] - idx_list[i] >= idx_threshold:
            new_list = idx_list[start : i + 1]
            start = i + 1
            new_list_dict = {
                "list": new_list,
                "length": len(new_list),
                "min": new_list[0],
                "max": new_list[-1]
            }
            new_lists.append(new_list_dict)
    return new_lists


def get_contact_related_contours(img, contact):
    def mark(contour, result, x, y, z):
        if contour[x, y, z] == result[x, y, z]:
            return
        else:
            result[x, y, z] = 1
            yz_iter(contour, result, x, y, z)

    def yz_iter(contour, result, x, y, z):

        mark(contour, result, x, min(y + 1, contour.shape[1] - 1), z)
        mark(contour, result, x, max(y - 1, 0), z)
        mark(contour, result, x, y, min(z + 1, contour.shape[2] - 1))
        mark(contour, result, x, y, min(z - 1, 0))

        mark(contour, result, x, min(y + 1, contour.shape[1] - 1), min(z + 1, contour.shape[2] - 1))
        mark(contour, result, x, min(y + 1, contour.shape[1] - 1), min(z - 1, 0))
        mark(contour, result, x, max(y - 1, 0), min(z + 1, contour.shape[2] - 1))
        mark(contour, result, x, max(y - 1, 0), min(z - 1, 0))
        # if x > 0:
        #     mark(contour, result, x - 1, y, z)
        # if y > 0:
        #     mark(contour, result, x, y - 1, z)
        # if x < contour.shape[0] - 1:
        #     mark(contour, result, x + 1, y, z)
        # if y < contour.shape[1] - 1:
        #     mark(contour, result, x, y + 1, z)
        # if x > 0 and y > 0:
        #     mark(contour, result, x - 1, y - 1, z)
        # if x < contour.shape[0] - 1 and y < contour.shape[1] - 1:
        #     mark(contour, result, x + 1, y + 1, z)
        # if x > 0 and y < contour.shape[1] - 1:
        #     mark(contour, result, x - 1, y + 1, z)
        # if y > 0 and x < contour.shape[0] - 1:
        #     mark(contour, result, x + 1, y - 1, z)

    result = np.zeros(contact.shape)

    indices = np.where(contact == 1)
    for i, j, k in zip(list(indices[0]), list(indices[1]), list(indices[2])):
        mark(img, result, i, j, k)
    return result


if __name__ == '__main__':

    target_only_filename = "artery_30_seg.nii.gz"
    contact_filename = "contact_30_seg.nii.gz"

    target = "artery"

    # get contour dict (contains contour and contact images)
    img_dict = get_contour_img(contact_filename)

    # get target-only one-hot mask image
    img_target = get_target_img(target_only_filename)
    print("image shape: " + str(img_target.shape))

    # get contact surface point in form of list
    contact_point_list = get_contact_point_list(img_dict, target, print_info=False)

    # get skeleton information: point_list - cluster points ; path_list - points in paths
    point_list, path_list = get_skeleton_summary(target_only_filename, print_info=False)

    img_range = get_contact_related_contours(img_target, img_dict["contact_artery"])

    indices = np.where(img_range == 1)
    x_s = []
    y_s = []
    z_s = []
    for i, j, k in zip(list(indices[0]), list(indices[1]), list(indices[2])):
        x_s.append(i)
        y_s.append(j)
        z_s.append(k)
    x_min = min(x_s)
    x_max = max(x_s)
    y_min = min(y_s)
    y_max = max(y_s)
    z_min = min(z_s)
    z_max = max(z_s)

    # new_img = np.zeros(img_target.shape)
    # new_img[x_min: x_max + 1, y_min: y_max + 1, z_min: z_max + 1] = img_target[x_min: x_max + 1, y_min: y_max + 1, z_min: z_max + 1]
    #
    #
    # new_nii = sitk.GetImageFromArray(new_img.astype(np.uint8))
    #
    # sitk.WriteImage(new_nii, "test.nii.gz")

    # link contact surface points with skeleton path points
    contact_related_list = get_contact_related_list(contact_point_list, path_list, img_target, print_info=False)

    for path_dict in path_list:
        print("path id:" + str(path_dict["id"]))
        for idx, path_point in enumerate(path_dict["path"]):
            if x_min <= path_point[0] <= x_max and y_min <= path_point[1] <= y_max and z_min <= path_point[2] <= z_max:
                print("    point " + str(idx) + ": " + str(path_point))


# path id:42
#     point 14: (30, 100, 104)
#     point 15: (29, 101, 105)
#     point 16: (29, 102, 104)
#     point 17: (28, 103, 104)
#     point 18: (28, 104, 105)
#     point 19: (27, 105, 104)
#     point 20: (26, 106, 103)
#     point 21: (25, 107, 102)
#     point 22: (25, 108, 102)
#     point 23: (24, 109, 103)
#     point 24: (24, 110, 102)
#     point 25: (23, 111, 101)
#     point 26: (22, 112, 100)
#     point 27: (22, 113, 101)
#     point 28: (21, 114, 100)
#     point 29: (20, 115, 99)
#     point 30: (19, 116, 98)
#     point 31: (18, 117, 97)
#     point 32: (17, 118, 96)
#     point 33: (16, 119, 95)
#     point 34: (15, 120, 94)
# path id:43
#     point 14: (30, 100, 104)
#     point 15: (29, 101, 105)
#     point 16: (29, 102, 104)
#     point 17: (28, 103, 104)
#     point 18: (28, 104, 105)
#     point 19: (27, 105, 104)
#     point 20: (26, 106, 103)
#     point 21: (25, 107, 102)
#     point 22: (25, 108, 102)
#     point 23: (24, 109, 103)
#     point 24: (24, 110, 102)
#     point 25: (23, 111, 101)
#     point 26: (22, 112, 100)
#     point 27: (22, 113, 101)
#     point 28: (21, 114, 100)
#     point 29: (20, 115, 99)
#     point 30: (19, 116, 98)
#     point 31: (18, 117, 97)
#     point 32: (17, 118, 96)
#     point 33: (16, 119, 95)
#     point 34: (15, 120, 94)
