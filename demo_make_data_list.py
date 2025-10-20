import os

data_list_file = 'data_list_vein_part.txt'
data_root = 'data_2_part_vein_remerge-0727'
data_list = []
for data_name in os.listdir(data_root):
    data_list.append(os.path.join(data_root, data_name))
txt_file = open(data_list_file, 'w', encoding='utf-8')
txt_file.writelines('\n'.join(data_list))
txt_file.close()