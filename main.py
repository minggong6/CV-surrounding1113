import logging
import os
import time

import xlrd2
import xlwt
import tqdm
import toolkit_main as tkm
import toolkit_3D as tk3
from SkeletonAnalysis import skeleton_analysis
from contact2D import calculate_2D_contact
from contact3D import calculate_3D_contact

if __name__ == '__main__':
    # region <------------------------- SET PARAMETERS ------------------------->
    # dataset_path = "newdata_1_preprocess"
    # dataset_path = "data_1_preprocess"
    # data_list_path = "data_list(all_processed_data).txt"
    # data_list_path = "data_list(all_wy_processed).txt"
    data_list_path = "data_list(0630new).txt"
    data_contact_path = "data_2.2_contact_z"
    timer = time.strftime("-%Y%m%d-%H%M%S", time.localtime())
    brief_path = "resources/brief_PDAC.xlsx"
    use_brief = False
    do_generating_contact = True
    do_2D_contact_analysis = False
    axis = 'z'
    do_3D_contact_analysis = False
    do_skeleton_analysis = True

    result_xls_path = "results/result" + timer + "(" + axis + ").xls"
    # endregion <------------------------- SET PARAMETERS ------------------------->

    if not os.path.exists(result_xls_path):
        with open(result_xls_path, mode='w', encoding='utf-8') as ff:
            logging.info('"' + result_xls_path + '" does not exist, successfully created.')

    # Image files loading
    file_list = tkm.get_img_file_list(data_list_path)
    """
    file_dict = {
        "img_id": "1",
        "img_path": "data/1_seg.nii.gz",
        "img_contact_path": "data_contact_z/contact_1_seg.nii.gz"
    }
    """
    if use_brief:
        brief_list = tkm.get_brief(brief_path)

    # region <====== Excel Initialization ======>

    # New Excel
    book = xlwt.Workbook()
    # Initialize 2 sheets
    contact_2D_sheet = book.add_sheet('Contact 2D Analysis')
    contact_3D_sheet = book.add_sheet('Contact 3D Analysis')
    skeleton_sheet = book.add_sheet('Skeleton Analysis')

    # Inject titles
    contact_2D_title = ['Case ID',  # 0
                        'Case Status',  # 1
                        'Target',  # 2
                        'Case Max Ratio',  # 3
                        'Case Max Ratio Slice Num',  # 4
                        'Slice Number',  # 5
                        'Slice Max Ratio',  # 6
                        'Part Contour',  # 7
                        'Part Contact',  # 8
                        'Part Ratio',  # 9
                        'Part Area',  # 10
                        'Part Roundness',  # 11
                        ]
    for col, t in enumerate(contact_2D_title):
        contact_2D_sheet.write(0, col, t)

    contact_3D_title = ['Case ID',  # 0
                        'Case Status',  # 1
                        'Contact ID',  # 2
                        'Target',  # 3
                        'Z-axis Range',  # 4
                        'Contact Pixels',  # 5
                        'Contour Pixels',  # 6
                        'Contact Ratio'  # 7
                        ]
    for col, t in enumerate(contact_3D_title):
        contact_3D_sheet.write(0, col, t)

    skeleton_title = ['Case ID',  # 0
                      'Case Status',  # 1
                      'Target',  # 2
                      'Skeleton Path ID',  # 3
                      'Total Path Length',  # 4
                      'In-path Contact ID',  # 5
                      'Contact Length'  # 6
                      ]
    for col, t in enumerate(skeleton_title):
        skeleton_sheet.write(0, col, t)
    style_red = xlwt.easyxf('pattern: pattern solid, fore_colour red;')
    # endregion <====== Excel Initialization ======>

    # region <====== Generating Contour & Contact File ======>
    if do_generating_contact:
        for idx, file_dict in enumerate(tqdm.tqdm(file_list, desc="Generating Contact")):
            tkm.generate_contour_nii_3D(file_dict, data_contact_path, prefix="contact_", contour_thickness=1.5,
                                        contact_range=2, axis=axis)

    """
        file_dict["img_contact_path"]
        background - 0
        tumor - 1
        contour_vein - 2
        contact_vein - 3
        contour_artery - 4
        contact_artery - 5
    """
    # endregion <====== Generating Contour & Contact File ======>

    # region <====== 2D Contact Analysis ======>
    """
        slice_dict = {
            "slice_num": slice_num,
            "slice_result_list": slice_result_list,
            "slice_max_ratio": slice_max_ratio
        }
    """
    if do_2D_contact_analysis:
        row = 1
        # for file_dict in file_list:
        for idx, file_dict in enumerate(tqdm.tqdm(file_list, desc="2D Contact Analyzing")):
            for target in ["vein", "artery"]:

                contact_2D_result, max_ratio, max_slice = tkm.calculate_2D_contact(file_dict, target, contact_dir=data_contact_path,
                                                                                   size_threshold=30, axis=axis)
                if use_brief:
                    wrapping, deformation = tkm.get_brief_info(brief_list, file_dict["img_id"], target)

                # slice_result_dict = {
                #     "slice_num": -1,
                #     "slice_max_ratio": 0,
                #     "part_contour": 0,
                #     "part_contact": 0,
                #     "part_ratio": 0,
                #     "part_area": 0,
                #     "part_roundness": 0
                # }
                for slice_dict in contact_2D_result:
                    # Case ID
                    contact_2D_sheet.write(row, 0, file_dict["img_id"])
                    # Case Status
                    if use_brief:
                        contact_2D_sheet.write(row, 1, wrapping)
                    # Target
                    contact_2D_sheet.write(row, 2, target)
                    # Case Max Ratio
                    contact_2D_sheet.write(row, 3, float(max_ratio))
                    # Case Max Ratio Slice
                    contact_2D_sheet.write(row, 4, int(max_slice))
                    # Slice Number
                    contact_2D_sheet.write(row, 5, int(slice_dict["slice_num"]))
                    # Slice Max Ratio
                    ratio = float(slice_dict["slice_max_ratio"])
                    if ratio > 0.5:
                        contact_2D_sheet.write(row, 6, ratio, style_red)
                    else:
                        contact_2D_sheet.write(row, 6, ratio)
                    # Part Contour
                    contact_2D_sheet.write(row, 7, int(slice_dict["part_contour"]))
                    # Part Contact
                    contact_2D_sheet.write(row, 8, int(slice_dict["part_contact"]))
                    # Part Ratio
                    contact_2D_sheet.write(row, 9, float(slice_dict["part_ratio"]))
                    # Part Area
                    contact_2D_sheet.write(row, 10, int(slice_dict["part_area"]))
                    # Part Roundness
                    contact_2D_sheet.write(row, 11, float(slice_dict["part_roundness"]))

                    row += 1
    # endregion <====== 2D Contact Analysis ======>

    # region <====== 3D Contact Analysis ======>
    """
        contact_dict = {
            "seg_id": i + 1,
            "z_range": (a, b),
            "contact_pixels": total_contact,
            "contour_pixels": total_contour,
            "contact_ratio": ratio
        }
    """
    if do_3D_contact_analysis:
        row = 1
        for idx, file_dict in enumerate(tqdm.tqdm(file_list, desc="3D Contact Analyzing")):
            for target in ["vein", "artery"]:
                contact_3D_result = calculate_3D_contact(file_dict, target)
                for slice_dict in contact_3D_result:
                    # Case ID
                    contact_3D_sheet.write(row, 0, file_dict["img_id"])
                    # Case Status
                    contact_3D_sheet.write(row, 1, "")
                    # Contact ID
                    contact_3D_sheet.write(row, 2, slice_dict["seg_id"])
                    # Target
                    contact_3D_sheet.write(row, 3, target)
                    # Z-axis Range
                    contact_3D_sheet.write(row, 4, str(slice_dict["z_range"]))
                    # Contact Pixels
                    contact_3D_sheet.write(row, 5, int(slice_dict["contact_pixels"]))
                    # Contour Pixels
                    contact_3D_sheet.write(row, 6, int(slice_dict["contour_pixels"]))
                    # Contact Ratio
                    contact_3D_sheet.write(row, 7, float(slice_dict["contact_ratio"]))
                    row += 1
    # endregion <====== 3D Contact Analysis ======>

    # region <====== Skeleton Analysis ======>
    """
        skeleton_dict = {
            "path_id": path_id,
            "path_length": p["length"],
            "path_start_point": p["start_point"],
            "path_end_point": p["end_point"],
            "path": p["path"],
            "decrypted_list": decrypted_list
        }
            decrypted_dict = {
                "id": i + 1,
                "list": new_list,
                "length": new_list[-1] - new_list[0] + 1,
                "min": new_list[0],
                "max": new_list[-1]
            }
    """
    if do_skeleton_analysis:
        row = 1
        for idx, file_dict in enumerate(tqdm.tqdm(file_list, desc="Skeleton Analyzing")):
            for target in ["vein", "artery"]:
                skeleton_result = skeleton_analysis(file_dict, target, print_info=False)
                for skeleton_dict in skeleton_result:
                    for decrypted_dict in skeleton_dict["decrypted_list"]:
                        # Case ID
                        skeleton_sheet.write(row, 0, file_dict["img_id"])
                        # Case Status
                        skeleton_sheet.write(row, 1, "")
                        # Target
                        skeleton_sheet.write(row, 2, target)
                        # Skeleton Path ID
                        skeleton_sheet.write(row, 3, int(skeleton_dict["path_id"]))
                        # Total Path Length
                        skeleton_sheet.write(row, 4, int(skeleton_dict["path_length"]))
                        # In-path Contact ID
                        skeleton_sheet.write(row, 5, int(decrypted_dict["id"]))
                        # Contact Length
                        skeleton_sheet.write(row, 6, int(decrypted_dict["length"]))
                        row += 1
    # endregion <====== Skeleton Analysis ======>

    # Save result xls
    book.save(result_xls_path)
