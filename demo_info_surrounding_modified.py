import os.path
import toolkit_3D as tk3
import xlrd2
import xlwt


def get_surrounding_position(xls_path):
    brief_book = xlrd2.open_workbook(xls_path)
    # brief_sheet = brief_book.sheet_by_index(0)
    sheets = [(brief_book.sheet_by_name("x"), 'x'),
              (brief_book.sheet_by_name("y"), 'y'),
              (brief_book.sheet_by_name("z"), 'z')]

    case_dict = {}
    for sheet, axis in sheets:
        for row in range(1, sheet.nrows):
            case_id = str(sheet.cell_value(row, 0))
            target = str(sheet.cell_value(row, 2))
            ratio = float(sheet.cell_value(row, 9))
            position = str(sheet.cell_value(row, 12))
            position = tk3.unravel_coordinate(position)
            # print(position)
            # if axis == 'z':
            #     pass
            # elif axis == 'x':
            #     position = (position[1], position[0], position[2])
            #     # img = img.transpose(1, 0, 2)  # z <-> x
            # elif axis == 'y':
            #     position = (position[2], position[1], position[0])
            #     # img = img.transpose(2, 1, 0)  # z <-> y
            print(axis + ': ' + str(position))
            if case_id not in case_dict.keys():
                case_dict[case_id] = {}
                case_dict[case_id]["artery"] = {"x": [], "y": [], "z": []}
                case_dict[case_id]["vein"] = {"x": [], "y": [], "z": []}
            if position[0] >= 0:
                case_dict[case_id][target][axis].append((position, ratio))
    return case_dict


# xls_path = r"results/result_surrounding-20230608.xls"
xls_path = r"results/info_material_surrounding_0925.xls"
artery_part_path = r"data_2_part_artery_remerge-0727"
vein_part_path = r"data_2_part_vein_remerge-0727"
result_path = r"results\result_part_surrounding_0925.xls"

case_dict = get_surrounding_position(xls_path)

part_num = 10
book = xlwt.Workbook()
artery_sheet = book.add_sheet('artery')
artery_title = ['Case ID',
                '1-x', '1-y', '1-z', '1-max',
                '2-x', '2-y', '2-z', '2-max',
                '3-x', '3-y', '3-z', '3-max',
                '4-x', '4-y', '4-z', '4-max',
                '5-x', '5-y', '5-z', '5-max',
                '6-x', '6-y', '6-z', '6-max',
                '7-x', '7-y', '7-z', '7-max',
                '8-x', '8-y', '8-z', '8-max',
                '9-x', '9-y', '9-z', '9-max',
                '10-x', '10-y', '10-z', '10-max',
                ]
for col, t in enumerate(artery_title):
    artery_sheet.write(0, col, t)

vein_sheet = book.add_sheet('vein')
vein_title = ['Case ID',
              '1-x', '1-y', '1-z', '1-max',
              '2-x', '2-y', '2-z', '2-max',
              '3-x', '3-y', '3-z', '3-max',
              ]
for col, t in enumerate(vein_title):
    vein_sheet.write(0, col, t)

row = 1
for case_id in case_dict.keys():
    artery_sheet.write(row, 0, case_id)
    case_part_path = os.path.join(artery_part_path, case_id + "_artery.nii.gz")
    if not os.path.exists(case_part_path):
        row += 3
        continue

    value_list = []

    for target in ['artery', 'vein']:

        if target == 'vein':
            continue

        vol_bias = 0

        for axis in ['x', 'y', 'z']:
            img = tk3.get_any_nii(case_part_path, axis=axis)["img"]
            # print(case_id + ' ' + axis + ': ' + str(img.shape))
            value_list = []
            for vol in range(0, part_num):
                value_list.append([])

            surrounding_list = case_dict[case_id][target][axis]
            for coordinate, ratio in surrounding_list:
                # print('\t' + str(coordinate))
                value = int(img[coordinate])
                if 0 < value <= part_num:
                    value_list[value - 1].append(ratio)
                    print(case_id, coordinate, value, ratio)

            for vol in range(0, part_num):
                if len(value_list[vol]) == 0:
                    continue
                value_str = ""
                max_value = 0
                for value in value_list[vol]:
                    if max_value < value:
                        max_value = value
                    # value_str += str(round(value, 2))
                    # value_str += ", "
                print("max_value = " + str(max_value))
                artery_sheet.write(row, vol * 4 + vol_bias + 1, float(max_value))

            vol_bias += 1

    row += 1

row = 1
for case_id in case_dict.keys():
    vein_sheet.write(row, 0, case_id)
    case_part_path = os.path.join(vein_part_path, case_id + "_vein.nii.gz")
    if not os.path.exists(case_part_path):
        row += 3
        continue

    value_list = []

    for target in ['artery', 'vein']:

        if target == 'artery':
            continue

        vol_bias = 0

        for axis in ['x', 'y', 'z']:
            img = tk3.get_any_nii(case_part_path, axis=axis)["img"]
            # print(case_id + ' ' + axis + ': ' + str(img.shape))
            value_list = []
            for vol in range(0, part_num):
                value_list.append([])

            surrounding_list = case_dict[case_id][target][axis]
            for coordinate, ratio in surrounding_list:
                # print('\t' + str(coordinate))
                value = int(img[coordinate])
                if 0 < value <= part_num:
                    value_list[value - 1].append(ratio)
                    print(case_id, coordinate, value, ratio)

            for vol in range(0, part_num):
                if len(value_list[vol]) == 0:
                    continue
                value_str = ""
                max_value = 0
                for value in value_list[vol]:
                    if max_value < value:
                        max_value = value
                    # value_str += str(round(value, 2))
                    # value_str += ", "
                print("max_value = " + str(max_value))
                vein_sheet.write(row, vol * 4 + vol_bias + 1, float(max_value))

            vol_bias += 1

    row += 1

book.save(result_path)

# position_dict = {}
# for case_dict in case_list:
#     if case_dict["artery_info"] == "":
#         position_dict[case_dict["case_id"]] = None
#         continue
#     info_list = case_dict["artery_info"].strip().split("\n")
#     coordinate_list = []
#     for info in info_list:
#         coordinate_str = info.split("(")[1].split(")")[0]
#         coordinate_str_list = coordinate_str.split(',')
#         coordinate = (int(coordinate_str_list[0].strip()),
#                       int(coordinate_str_list[1].strip()),
#                       int(coordinate_str_list[2].strip()))
#         coordinate_list.append(coordinate)
#     if len(coordinate_list) > 0:
#         position_dict[case_dict["case_id"]] = coordinate_list
#     else:
#         position_dict[case_dict["case_id"]] = None
#
# print(position_dict)
#
# book = xlwt.Workbook()
# # Initialize 2 sheets
# sheet = book.add_sheet('artery')
#
# title = ['Case ID',  # 0
#          '1',  # 1
#          '2',  # 2
#          '3',  # 3
#          '4',  # 4
#          '5',  # 5
#          '6',  # 6
#          '7',  # 7
#          '8',  # 8
#          '9',  # 9
#          '10',  # 10
#          ]
# for col, t in enumerate(title):
#     sheet.write(0, col, t)
#
# row = 1
#
# for case_id in position_dict.keys():
#     if position_dict[case_id] is None:
#         sheet.write(row, 0, case_id)
#         row += 1
#         continue
#     case_part_path = os.path.join(part_path, case_id + "_artery.nii.gz")
#     if not os.path.exists(case_part_path):
#         sheet.write(row, 0, case_id)
#         row += 1
#         continue
#     img_dict = tk3.get_any_nii(case_part_path)
#     img = img_dict["img"]
#     img_info = img_dict["info"]
#
#     value_list = []
#     for coordinate in position_dict[case_id]:
#         value = int(img[coordinate])
#         if 0 < value <= 10:
#             if value not in value_list:
#                 value_list.append(value)
#     sheet.write(row, 0, case_id)
#     for value in value_list:
#         sheet.write(row, value, 1)
#
#     row += 1
#
# book.save(result_path)
