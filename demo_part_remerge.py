import logging
import os.path

import numpy as np
import toolkit_3D as tk3

target_dict = {
    'AO': 1,  # 腹主动脉
    'CA': 2,  # 腹腔干
    'LGA': 3,  # 胃左动脉
    'SA': 4,  # 脾动脉
    'RHA': 5,  # 肝右动脉
    'SMA': 6  # 肠系膜上动脉
}

contain_dict = {
    '0': {'AO': [],  # 腹主动脉
          'CA': [],  # 腹腔干
          'LGA': [],  # 胃左动脉
          'SA': [],  # 脾动脉
          'RHA': [],  # 肝右动脉
          'SMA': [],  # 肠系膜上动脉
          },
    '1': {'AO': [34, 35, 1],  # 腹主动脉
          'CA': [32],  # 腹腔干
          'LGA': [57, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2],  # 胃左动脉
          'SA': [19, 17, 18],  # 脾动脉
          'RHA': [31, 29, 20, 30, 25, 28, 27, 26, 21, 24, 23, 22],  # 肝右动脉
          'SMA': [39, 33, 37],  # 肠系膜上动脉
          },
}


def get_contain(img_id):
    if img_id in contain_dict.keys():
        return contain_dict[img_id]
    else:
        logging.error('No contain info !')
        exit(-1)


def remerge(img, contain_dict):
    remerged_img = np.zeros(img.shape)
    for target in target_dict.keys():
        target_id = target_dict[target]
        target_contain_list = contain_dict[target]
        for contain_id in target_contain_list:
            remerged_img += np.where(img == contain_id, target_id, 0)
            img = np.where(img == contain_id, 0, img)
    max_old_id = np.max(img)
    new_id = 7
    if max_old_id > 0:
        for old_id in range(1, max_old_id + 1):
            if np.sum(img == old_id) > 0:
                remerged_img += np.where(img == old_id, new_id, 0)
                new_id += 1
    return remerged_img


if __name__ == '__main__':
    data_dir = 'data_2_part_artery'
    save_dir = 'data_2_part_artery_remerge'

    img_id = '1'
    img_name = img_id + '_artery.nii.gz'
    img_path = os.path.join(data_dir, img_name)
    img_dict = tk3.get_any_nii(img_path)
    img, img_info = img_dict['img'], img_dict['info']

    contain_dict = get_contain(img_id)

    remerged_img = remerge(img, contain_dict)

    save_path = os.path.join(save_dir, img_name)
    tk3.save_nii(remerged_img, save_path, img_info)