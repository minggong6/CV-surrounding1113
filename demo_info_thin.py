import os.path
import toolkit_3D as tk3
import xlrd2
import xlwt


def get_thin_position(xls_path):
    brief_book = xlrd2.open_workbook(xls_path)
    # brief_sheet = brief_book.sheet_by_index(0)
    brief_sheet = brief_book.sheet_by_name("sheet1")

    case_list = []
    for row in range(2, brief_sheet.nrows):
        case_id = str(brief_sheet.cell_value(row, 0))
        artery_info = str(brief_sheet.cell_value(row, 1))
        vein_info = str(brief_sheet.cell_value(row, 4))
        case_dict = {
            "case_id": case_id,
            "artery_info": artery_info,
            "vein_info": vein_info
        }
        case_list.append(case_dict)
    return case_list


# xls_path = r"D:\Notes\胰腺任务\analysis0420_thin.xlsx"
# xls_path = r"results/result_thin-20230917-164120.xls"
# xls_path = r"results/result_thin-20230926.xls"
# result_path = r"results\result_part_thin_0926.xls"
xls_path = r"results/result_thin-20230927-132736(test).xls"
result_path = r"results/result_part_thin_0926(avg-ALL).xls"

part_path = r"data_2_part_artery_remerge-0727"
case_list = get_thin_position(xls_path)

book = xlwt.Workbook()
# Initialize 2 sheets
sheet = book.add_sheet('artery')

title = ['Case ID',  # 0
         '1',  # 1
         '2',  # 2
         '3',  # 3
         '4',  # 4
         '5',  # 5
         '6',  # 6
         '7',  # 7
         '8',  # 8
         '9',  # 9
         '10',  # 10
         ]
for col, t in enumerate(title):
    sheet.write(0, col, t)

position_dict = {}
for case_dict in case_list:
    if case_dict["artery_info"] == "":
        position_dict[case_dict["case_id"]] = None
        continue
    info_list = case_dict["artery_info"].strip().split("\n")
    coordinate_list = []
    for info in info_list:
        coordinate_str = info.split("(")[1].split(")")[0]
        coordinate_str_list = coordinate_str.split(',')
        coordinate = (int(coordinate_str_list[0].strip()),
                      int(coordinate_str_list[1].strip()),
                      int(coordinate_str_list[2].strip()))
        coordinate_list.append(coordinate)
    if len(coordinate_list) > 0:
        position_dict[case_dict["case_id"]] = coordinate_list
    else:
        position_dict[case_dict["case_id"]] = None

print(position_dict)

row = 1

for case_id in position_dict.keys():
    if position_dict[case_id] is None:
        sheet.write(row, 0, int(case_id))
        row += 1
        continue
    case_part_path = os.path.join(part_path, case_id + "_artery.nii.gz")
    if not os.path.exists(case_part_path):
        sheet.write(row, 0, case_id)
        row += 1
        continue
    img_dict = tk3.get_any_nii(case_part_path)
    img = img_dict["img"]
    img_info = img_dict["info"]

    value_list = []
    for coordinate in position_dict[case_id]:
        value = int(img[coordinate])
        if 0 < value <= 10:
            if value not in value_list:
                value_list.append(value)
    sheet.write(row, 0, int(case_id))
    for value in value_list:
        sheet.write(row, value, 1)

    row += 1

part_path = r"data_2_part_vein_remerge-0727"
sheet = book.add_sheet('vein')
title = ['Case ID',  # 0
         '1',  # 1
         '2',  # 2
         '3',  # 3
         ]
for col, t in enumerate(title):
    sheet.write(0, col, t)

position_dict = {}
for case_dict in case_list:
    if case_dict["vein_info"] == "":
        position_dict[case_dict["case_id"]] = None
        continue
    info_list = case_dict["vein_info"].strip().split("\n")
    coordinate_list = []
    for info in info_list:
        coordinate_str = info.split("(")[1].split(")")[0]
        coordinate_str_list = coordinate_str.split(',')
        coordinate = (int(coordinate_str_list[0].strip()),
                      int(coordinate_str_list[1].strip()),
                      int(coordinate_str_list[2].strip()))
        coordinate_list.append(coordinate)
    if len(coordinate_list) > 0:
        position_dict[case_dict["case_id"]] = coordinate_list
    else:
        position_dict[case_dict["case_id"]] = None

print(position_dict)

row = 1

for case_id in position_dict.keys():
    if position_dict[case_id] is None:
        sheet.write(row, 0, int(case_id))
        row += 1
        continue
    case_part_path = os.path.join(part_path, case_id + "_vein.nii.gz")
    if not os.path.exists(case_part_path):
        sheet.write(row, 0, int(case_id))
        row += 1
        continue
    img_dict = tk3.get_any_nii(case_part_path)
    img = img_dict["img"]
    img_info = img_dict["info"]

    value_list = []
    for coordinate in position_dict[case_id]:
        value = int(img[coordinate])
        if 0 < value <= 10:
            if value not in value_list:
                value_list.append(value)
    sheet.write(row, 0, case_id)
    for value in value_list:
        sheet.write(row, value, 1)

    row += 1


book.save(result_path)
