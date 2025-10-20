import os
import SimpleITK as sitk
import nibabel as nib


def generate_nii_list(dir_list, txt_path=None):
    text_list = []
    for dir in dir_list:
        for _, _, files in os.walk(dir):
            for file in files:
                path = os.path.join(dir, file)
                if txt_path is not None:
                    text_list.append(str(path) + '\n')
                else:
                    text_list.append(str(path))
    if txt_path is not None:
        text = open(txt_path, mode='w')
        text.writelines(text_list)
        print(str(len(text_list)) + ' cases of data successfully generated.')
    return text_list


def read_nii_list(data_list_path):
    data_list = []
    file = open(data_list_path, 'r')
    file_data = file.readlines()
    for row in file_data:
        data_item = row.strip()
        if data_item[0] == '#':
            continue
        data_list.append(row.strip())
    print(str(len(data_list)) + ' cases of data successfully read.')
    return data_list


def data_check(txt_path):
    nii_list = read_nii_list(txt_path)
    total_case = 0
    correct_case = 0
    error_case = 0
    for nii_path in nii_list:
        try:
            image = sitk.ReadImage(nii_path)
            image = sitk.GetArrayFromImage(image)
        except Exception as e:
            print('"' + nii_path + '" case error: ' + str(e))
            error_case += 1
        else:
            correct_case += 1
        finally:
            total_case += 1

    print('Data status:')
    print('\tTotal:' + str(total_case))
    print('\tCorrect:' + str(correct_case))
    print('\tError:' + str(error_case))


def rename_data_pack2(data_p2_path):

    dir_list = [data_p2_path, ]
    nii_list = generate_nii_list(dir_list)
    for nii_path in nii_list:
        nii_name = os.path.basename(nii_path)
        nii_dirname = os.path.dirname(nii_path)
        status = nii_name.split('_')[0]
        id = nii_name.split('_')[1].split('.')[0]
        new_nii_path = os.path.join(nii_dirname, id + '_' + status + '.nii.gz')
        try:
            os.rename(nii_path, new_nii_path)

        except Exception as e:
            print('Rename nii fail: ' + str(e))
        else:
            print('Rename nii from "' + nii_path + '" to "' + new_nii_path + '"')


def resave_nii(nii_path):
    data = nib.load(nii_path)
    q_form = data.get_qform()
    data.set_qform(q_form)
    s_from = data.get_sform()
    data.set_sform(s_from)
    nib.save(data, nii_path)


if __name__ == '__main__':
    # data_p2_path = 'data_p2.1'
    # rename_data_pack2(data_p2_path)

    # dir_list = ['data_p1_preprocess', 'data_p2_preprocess']
    # dir_list = ['data_p2.2_preprocess', ]
    # txt_path = 'data_list(all_wy_processed).txt'
    # generate_nii_list(dir_list, txt_path)
    #
    # data_check(txt_path)
    generate_nii_list(["data_p2.4", ], txt_path="data_list(final87).txt")

