import tqdm

import toolkit_3D as tk3
import toolkit_main as tkm
import SkeletonAnalysis as skele
from graph_v7 import get_skeleton_summary

# region <------------------------- SET PARAMETERS ------------------------->
from toolkit_skeleton import Skeleton

dataset_list_path = "data_list(all_wy_processed)1.txt"
part_artery = True
part_vein = False
# endregion <------------------------- SET PARAMETERS ------------------------->

# Image files loading
# file_list = tkm.get_img_file_list(dataset_list_path)

img_id = '49'
file_dict = {
    "img_id": img_id,
    "img_path": 'data_p2.3_preprocess\\' + img_id + '_pre.nii.gz',
    "img_contact_path": None
}

# img_id = '83'
# file_dict = {
#     "img_id": img_id,
#     "img_path": 'data_p2.3_preprocess\\' + img_id + '_pre.nii.gz',
#     "img_contact_path": None
# }

file_list = [file_dict, ]

vein_empty_list = []

'''
(x, y, z) -> x z
               y
'''

vein_cross_position_dict = {
    '5': (47, 108, 96),
    '10': (39, 115, 108),
    '13': (42, 129, 124),
    '15': (54, 122, 136),
    '19': (79, 111, 124),
    '20': (74, 101, 96),
    '29b': (60, 128, 107),
    '40': (76, 109, 168),
    '43': (41, 126, 115),
    '46': (51, 109, 96),
    '48': (40, 112, 101),
    '49': (33, 113, 101),
    '57': (80, 97, 136),
    '58': (59, 116, 104),
    '60': (45, 105, 101),
    '61': (45, 136, 96),
    '63': (39, 98, 132),
    '66': (52, 124, 109),
    '68': (47, 126, 112),
    '82b': (54, 114, 109),
    '86': (48, 106, 114),
    '101': (58, 82, 96),  # Need extra correction
    '102': (68, 122, 90),
    '107': (38, 118, 87),
    '116': (37, 108, 101),
    '118': (59, 117, 112),
    '121': (62, 114, 117),
    '123': (43, 104, 121)
}

artery_lists_dict = {
    '0': ([], []),
    '13': ([(42, 102, 136)], []),
    # '14': ([(24, 99, 112), (35, 89, 112)], [(4, 80, 116)]),
    '14': ([(74, 77, 149)], [(33, 70, 150), (81, 41, 142)]),
    '29a': ([(50, 82, 92), ], [(26, 86, 101)]),
    '29b': ([(57, 75, 98)], [(37, 80, 110)]),
    '31': ([(44, 96, 131)], [(33, 73, 135)]),
    '32': ([], []),
    '40': ([(48, 92, 132)], [(55, 83, 168)]),
    '42': ([(46, 92, 132)], [(31, 81, 116)]),
    '49': ([(31, 108, 110)], [(1, 94, 110)]),  # Problem
    '55': ([(43, 76, 104)], [(26, 75, 119)]),
    '57': ([(73, 93, 149)], [(50, 83, 131)]),
    '69': ([(49, 105, 107)], [(35, 89, 103)]),
    '72': ([], []),
    '79': ([(6, 90, 81), (1, 82, 129)], []),
    '83': ([(45, 95, 102)], [(5, 118, 86), (58, 86, 83)]),  # Problem
    '113': ([(33, 96, 112)], [(51, 62, 124), (18, 84, 106)]),
    '117': ([(51, 93, 108)], [(33, 78, 107)]),
}

'''
Unpartable cases: 12 14 15 25 27 29a 29b 30 32 35 50 67 68 79 80 83 106 123
'''


def get_vein_cross_position(img_id):
    if img_id in vein_cross_position_dict.keys():
        return vein_cross_position_dict[img_id]


def get_artery_lists(img_id):
    if img_id in artery_lists_dict.keys():
        return artery_lists_dict[img_id]


for file_dict in file_list:

    img_dict = tk3.get_nii(file_dict["img_path"])
    img_tumor = img_dict["tumor"]
    img_info = img_dict["info"]

    shape = img_tumor.shape

    if part_vein:

        target = 'vein'
        skeleton = Skeleton(img_dict[target])
        skeleton.process_vein_part(cross_position=get_vein_cross_position(file_dict["img_id"]))
        if skeleton.y_path_graph is None:
            vein_empty_list.append(file_dict["img_id"])
        else:
            tk3.save_nii(skeleton.part_graph, "data_2_part_vein/" + file_dict["img_id"] + "_" + target + ".nii.gz",
                         img_info)

    if part_artery:
        target = 'artery'
        skeleton = Skeleton(img_dict[target])
        # skeleton.process_artery_part()

        ignore_list, contain_list = get_artery_lists(file_dict["img_id"])

        skeleton.process_trunk_part(ignore_list=ignore_list, contain_list=contain_list)
        # skeleton.process_normal_part()
        tk3.save_nii(skeleton.part_graph, "data_2_part_artery_remerge-0727/" + file_dict["img_id"] + "_" + target + "_.nii.gz",
                     img_info)
        print('Saving ' + file_dict["img_id"])

print('vein_empty_list: ' + str(vein_empty_list))
