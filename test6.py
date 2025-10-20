import os

file_list_0 = os.listdir('data_p2.3_preprocess')
file_list_1 = os.listdir('data_2_part_artery_remerge-0727')
file_list_2 = os.listdir('data_2_part_vein_remerge-0727')

for idx in range(0, len(file_list_0)):
    file_list_0[idx] = file_list_0[idx].split('_')[0]
for idx in range(0, len(file_list_1)):
    file_list_1[idx] = file_list_1[idx].split('_')[0]
for idx in range(0, len(file_list_2)):
    file_list_2[idx] = file_list_2[idx].split('_')[0]

print('not in file1:')
for f0 in file_list_0:
    if f0 not in file_list_1:
        print(f0)

print('not in file2:')
for f0 in file_list_0:
    if f0 not in file_list_2:
        print(f0)