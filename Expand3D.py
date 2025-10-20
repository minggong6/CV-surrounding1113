import nibabel as nib
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import sys
import SimpleITK as sitk


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


def expand_3D(img, level=1):
    expand_img = np.array(img, copy=True)
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
            if i + a > img.shape[0] - 1 or i + a < 0 \
                    or j + b > img.shape[1] - 1 or j + b < 0 \
                    or k + c > img.shape[2] - 1 or k + c < 0:
                continue
            expand_img[i + a][j + b][k + c] = 1
    return expand_img


if __name__ == '__main__':
    filename = "2_thinning.nii.gz"

    result_dict = get_nii(filename, rotate=False)

    img = result_dict["origin"]

    img = expand_3D(img, level=2)

    new_nii = nib.Nifti1Image(img, result_dict["affine"], result_dict["header"])
    nib.save(new_nii, "expand_" + filename)
