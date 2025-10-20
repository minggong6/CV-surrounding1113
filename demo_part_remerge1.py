import logging
import os.path
from scipy import ndimage
import toolkit_main as tkm

import numpy as np
import toolkit_3D as tk3


def remerge(img, contain_dict, target_dict):
    bias = 10
    remerged_img = img.copy()
    remerged_img += bias
    remerged_img = np.where(remerged_img == bias, 0, remerged_img)

    target_id_list = []
    main_id = 1
    for target in target_dict.keys():
        target_id = target_dict[target]
        target_id_list.append(target_id)
        target_contain_list = contain_dict[target]
        for contain_id in target_contain_list:
            remerged_img = np.where(remerged_img == contain_id + bias, target_id, remerged_img)

    tk3.save_nii(remerged_img, 'test00.nii.gz', img_info)

    for target_id in target_id_list:
        if target_id == main_id:
            continue
        if np.sum(remerged_img == target_id) <= 0:
            continue
        else:
            mask_num = 1
            while mask_num > 0:
                print(mask_num)
                next_mask = ndimage.distance_transform_edt(np.where(remerged_img == target_id, 0, 1))
                next_mask = np.where(next_mask < 2, 1, 0)
                next_mask = np.multiply(next_mask, np.where(remerged_img <= bias, 0, 1))
                mask_num = np.sum(next_mask)
                next_graph = next_mask * target_id
                remerged_img = np.multiply(remerged_img, np.where(next_mask > 0, 0, 1)) + next_graph
    return remerged_img


if __name__ == '__main__':
    data_dir = 'data_2_part_artery_remerge-0727'
    save_dir = 'data_2_part_artery_remerge-0727'

    # Confused Data: 21, 50, 107, 113, 121, 123
    # Unable Data: 32, 72, 49

    img_id = '49'

    contain = tkm.Contain()

    img_name = img_id + '_artery.nii.gz'
    img_path = os.path.join(data_dir, img_name)
    img_dict = tk3.get_any_nii(os.path.join(save_dir, img_id + '_artery_.nii.gz'))
    img, img_info = img_dict['img'], img_dict['info']

    remerged_img = remerge(img, contain.get_contain(img_id), contain.target_dict)

    save_path = os.path.join(save_dir, img_id + '_artery__.nii.gz')
    tk3.save_nii(remerged_img, save_path, img_info)
