import os.path
import toolkit_3D as tk3
import xlrd2
import xlwt


def get_thin_full_info(xls_path):
    brief_book = xlrd2.open_workbook(xls_path)
    # brief_sheet = brief_book.sheet_by_index(0)
    brief_sheet = brief_book.sheet_by_name("sheet1")

    case_list = []
    for row in range(2, brief_sheet.nrows):
        case_id = int(brief_sheet.cell_value(row, 0))
        artery_info = str(brief_sheet.cell_value(row, 1))
        vein_info = str(brief_sheet.cell_value(row, 4))
        case_dict = {
            "case_id": case_id,
            "artery": artery_info,
            "vein": vein_info
        }
        case_list.append(case_dict)
    return case_list


# xls_path = r"D:\Notes\胰腺任务\analysis0420_thin.xlsx"
# xls_path = r"results/result_thin-20230803-153953_full_info.xls"
xls_path = r"results/result_thin-20230922-114231_full_info.xls"
part_paths = {
    'artery': r"data_2_part_artery_remerge-0727",
    'vein': r"data_2_part_vein_remerge-0727"
}
result_path = r"results/result_part_thin_0922_full_info.xls"

case_list = get_thin_full_info(xls_path)

book = xlwt.Workbook()
# Initialize 2 sheets
sheet = book.add_sheet('sheet1')

title = ['Case ID',  # 0
         'target',  # 1
         'branch',  # 2
         'coordinate',  # 3
         'radius',  # 4
         'radius_dist',  # 5
         'vessel_tumor_dist',  # 6
         ]
for col, t in enumerate(title):
    sheet.write(0, col, t)

full_info = {'artery': [], 'vein': []}

for case_dict in case_list:
    case_id = case_dict['case_id']
    if case_id != '25':
        continue
    for target in ['artery', 'vein']:
        if case_dict[target] == "":
            point_info_dict = {
                'case_id': case_id,
                'coordinate': None,
                'radius': None,
                'vessel_tumor_dist': None
            }
            full_info[target].append(point_info_dict)
            continue
        info_str = case_dict[target]
        point_list = info_str.strip().split(";")

        for point_info_str in point_list:
            if point_info_str == '':
                continue
            point_info_list = point_info_str.split("&")
            assert len(point_info_list) == 4, f'len(point_info_list)={len(point_info_list)}, {point_info_str}'

            coordinate_str = point_info_list[0]
            radius_str = point_info_list[1]
            radius_dist_str = point_info_list[2]
            vessel_tumor_dist_str = point_info_list[3]

            coordinate_str = coordinate_str.split("(")[1].split(")")[0]
            coordinate_str_list = coordinate_str.split(',')
            coordinate = (int(coordinate_str_list[0].strip()),
                          int(coordinate_str_list[1].strip()),
                          int(coordinate_str_list[2].strip()))

            radius = float(radius_str)
            radius_dist = float(radius_dist_str)
            vessel_tumor_dist = float(vessel_tumor_dist_str)

            point_info_dict = {
                'case_id': case_id,
                'coordinate': coordinate,
                'radius': radius,
                'radius_dist': radius_dist,
                'vessel_tumor_dist': vessel_tumor_dist
            }
            full_info[target].append(point_info_dict)


row = 1

empty_list = {'artery': [], 'vein': []}
exist_list = {'artery': [], 'vein': []}

for target in ['artery', 'vein']:
    for point_info_dict in full_info[target]:

        case_id = point_info_dict['case_id']
        if point_info_dict['coordinate'] is None:
            if case_id in empty_list[target] or case_id in exist_list[target]:
                continue
            else:
                empty_list[target].append(case_id)
                continue
        else:
            if case_id in empty_list:
                empty_list[target].remove(case_id)
            if case_id in exist_list:
                pass
            else:
                exist_list[target].append(case_id)



        case_part_path = os.path.join(part_paths[target], str(case_id) + '_' + target + '.nii.gz')
        if not os.path.exists(case_part_path):
            continue

        img_dict = tk3.get_any_nii(case_part_path)
        img = img_dict["img"]
        img_info = img_dict["info"]

        coordinate = point_info_dict['coordinate']
        branch = int(img[coordinate])
        write_info = True
        if target == 'artery':
            assert 0 <= branch <= 10, f'branch={branch}'
            if branch == 2:
                sheet.write(row, 2, 'CA')
            elif branch == 3:
                sheet.write(row, 2, 'CHA')
            elif branch == 6:
                sheet.write(row, 2, 'SMA')
            else:
                write_info = False
        else:
            assert 0 <= branch <= 3
            if branch == 1:
                sheet.write(row, 2, 'PV')
            elif branch == 2:
                sheet.write(row, 2, 'SMV')
            else:
                write_info = False
        if write_info:
            sheet.write(row, 0, case_id)
            sheet.write(row, 1, target)
            sheet.write(row, 3, str(point_info_dict['coordinate']))
            sheet.write(row, 4, point_info_dict['radius'])
            sheet.write(row, 5, point_info_dict['radius_dist'])
            sheet.write(row, 6, point_info_dict['vessel_tumor_dist'])
            row += 1

for target in ['artery', 'vein']:
    for case_id in empty_list[target]:
        sheet.write(row, 0, case_id)
        sheet.write(row, 1, target)
        row += 1

book.save(result_path)
