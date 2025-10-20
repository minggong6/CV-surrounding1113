import os

data_dir = "data_p2.2_preprocess"

with open("data_list(0630new).txt", 'w', encoding="utf-8") as f:
    data_str = ""
    file_list = os.listdir(data_dir)
    for file in file_list:
        file_path = os.path.join(data_dir, file)
        if data_str == "":
            data_str += str(file_path)
        else:
            data_str += "\n"
            data_str += str(file_path)
    f.write(data_str)