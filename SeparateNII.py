import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib


def get_nii(filename, rotate=True):
    # read nii 3D data
    nii = nib.load(filename)
    img = nii.get_fdata().astype(np.float32)

    if rotate:
        # rotate 90 to fit itk-snap
        img = np.rot90(img, k=1, axes=(0, 1))

    # separate labels
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


def get_located_pixel(img, x, y, z, value_overflow=1, value_within=None):
    shape = img.shape
    if x < 0 or x >= shape[0]:
        return value_overflow
    elif y < 0 or y >= shape[1]:
        return value_overflow
    elif z < 0 or z >= shape[2]:
        return value_overflow
    else:
        if value_within is None:
            return img[x, y, z]
        else:
            return value_within


def sum_surrounding(img, x, y, z, cubic_radis=2):
    sum = 0
    fix_range_list = get_fix_range_list(cubic_radis)
    voxel_num = len(fix_range_list)
    for (a, b, c) in fix_range_list:
        sum += get_located_pixel(img, x + a, y + b, z + c)
    return voxel_num, sum


def get_fix_range_list(cubic_radis):
    threshold = cubic_radis ^ 2 + 2
    fix_range_list = []
    for i in range(-2, 3):
        for j in range(-2, 3):
            for k in range(-2, 3):
                if i ^ 2 + j ^ 2 + k ^ 2 > threshold:
                    continue
                else:
                    fix_range_list.append((i, j, k))
    return fix_range_list


if __name__ == '__main__':
    # <------------------------- SET PARAMETERS ------------------------->
    # filename = "2_seg.nii.gz"
    # filename = "3_seg.nii.gz"
    filename = "30_seg.nii.gz"
    # filename = "32_seg.nii.gz"
    # filename = "35_seg.nii.gz"

    # target = "vein"
    target = "artery"

    # To get rid of hollow points inside the target
    fix_hollow = True

    # The range in which we detect hollow points
    cubic_radis = 2
    # <------------------------- SET PARAMETERS ------------------------->
    result_dict = get_nii(filename, rotate=False)

    img = result_dict[target]
    shape = img.shape

    if fix_hollow:

        total_fix = 0
        for i in range(0, shape[0]):
            for j in range(0, shape[1]):
                for k in range(0, shape[2]):
                    if img[i, j, k] == 1:
                        continue
                    voxel_num, sum = sum_surrounding(img, i, j, k, cubic_radis)

                    if sum >= voxel_num * 0.8:
                        img[i, j, k] = 1
                        total_fix += 1
                        print(f"[{i}, {j}, {k}] - voxel num = {voxel_num}, sum = {sum}, total fix = {total_fix}")

    new_nii = nib.Nifti1Image(img, result_dict["affine"], result_dict["header"])
    nib.save(new_nii, target + "_" + filename)
