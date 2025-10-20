import itk
import numpy as np
# python -m pip install --upgrade pip
# python -m pip install itk-thickness3d

def OutputTXT(filePath, list_centerLinePoint):
    with open(filePath, "w+") as fout:
        fout.writelines(
            [" ".join([str(i) for i in centerLinePoint]) + "\n" for centerLinePoint in list_centerLinePoint])


def Collect26Neighbors(image, p_index):
    offset = [
        [-1, -1, 1], [0, -1, 1], [1, -1, 1], [-1, -1, 0], [0, -1, 0], [1, -1, 0], [-1, -1, -1], [0, -1, -1],
        [1, -1, -1],
        [-1, 0, 1], [0, 0, 1], [1, 0, 1], [-1, 0, 0], [1, 0, 0], [-1, 0, -1], [0, 0, -1], [1, 0, -1],
        [-1, 1, 1], [0, 1, 1], [1, 1, 1], [-1, 1, 0], [0, 1, 0], [1, 1, 0], [-1, 1, -1], [0, 1, -1], [1, 1, -1]]
    Np = [1 if image.GetPixel(tuple([p_index[i] + offset[l][i] for i in range(3)])) == 1 else 0 for l in range(26)]
    return Np


def ReturnId(p_index, list_centerLinePoint):
    # [(id, x, y, z, parentId, leftChildId, rightChildId),...]

    id = None
    for i in range(len(list_centerLinePoint)):
        if list_centerLinePoint[i][1:4] == list(p_index):
            id = list_centerLinePoint[i][0]
            break
    return id


def ReturnIndex(j, p_index):
    offset = [
        [-1, -1, 1], [0, -1, 1], [1, -1, 1], [-1, -1, 0], [0, -1, 0], [1, -1, 0], [-1, -1, -1], [0, -1, -1],
        [1, -1, -1],
        [-1, 0, 1], [0, 0, 1], [1, 0, 1], [-1, 0, 0], [1, 0, 0], [-1, 0, -1], [0, 0, -1], [1, 0, -1],
        [-1, 1, 1], [0, 1, 1], [1, 1, 1], [-1, 1, 0], [0, 1, 0], [1, 1, 0], [-1, 1, -1], [0, 1, -1], [1, 1, -1]]
    re_index = [p_index[i] + offset[j][i] for i in range(3)]
    return re_index


def ReturnJx(Np, count=1):
    ct = 0
    j = []
    for i in range(26):
        if Np[i] == 1:
            j.append(i)
            ct += 1
        if count == ct:
            break
    return j[0] if count == 1 else j

def MarkAllPoints(image):
    size = image.GetLargestPossibleRegion().GetSize()
    id_ini = 0
    list_centerline_vec = []
    for i in range(size[0] - 1):
        for j in range(size[1] - 1):
            for k in range(size[2] - 1):
                index = (i, j, k)
                value = image.GetPixel(index)
                if value == 0:
                    continue
                else:
                    list_centerline_vec.append([id_ini, i, j, k])  # [(id, x, y, z),...]
                    id_ini = id_ini + 1
    return list_centerline_vec


def GenerateTree(image):
    centerline_vec = MarkAllPoints(image)
    for i in range(len(centerline_vec)):
        p_index = centerline_vec[i][1:4]
        Np = Collect26Neighbors(image, p_index)
        sum = 0
        for j in range(26):
            sum += Np[j]
        if sum == 0:
            centerline_vec[i].extend([-1, -1, -1])
        #  判断是端点的情况
        if sum == 1:
            j = ReturnJx(Np, count=1)
            if j <= 12:
                parent_index = ReturnIndex(j, p_index)
                centerline_vec[i].extend([ReturnId(parent_index, centerline_vec), -1, -1])
            # 如果只有一个childId一律设置成左childId
            if j >= 13:
                leftChild_index = ReturnIndex(j, p_index)
                centerline_vec[i].extend([-1, ReturnId(leftChild_index, centerline_vec), -1])
        if sum == 2:
            j = ReturnJx(Np, count=2)
            parent_index = ReturnIndex(j[0], p_index)
            leftChild_index = ReturnIndex(j[1], p_index)
            centerline_vec[i].extend(
                [ReturnId(parent_index, centerline_vec), ReturnId(leftChild_index, centerline_vec), -1])
        if sum == 3:
            j = ReturnJx(Np, count=3)
            parent_index = ReturnIndex(j[0], p_index)
            leftChild_index = ReturnIndex(j[1], p_index)
            rightChild_index = ReturnIndex(j[2], p_index)
            centerline_vec[i].extend([ReturnId(parent_index, centerline_vec), ReturnId(leftChild_index, centerline_vec),
                                      ReturnId(rightChild_index, centerline_vec)])
    return centerline_vec


if __name__ == '__main__':
    input_filename = "sep_vein_2_seg.nii.gz"
    img = itk.imread(input_filename)

    print("Skeletonizing...")
    thinning_map = itk.BinaryThinningImageFilter3D.New(img)
    print("Skeletonization complete")
    itk.imwrite(thinning_map, "2_thinning.nii.gz")
    image = thinning_map.GetOutput()
    centerline_vec_final = GenerateTree(image)
    path = "CenterlinePoints.txt"
    OutputTXT(path, centerline_vec_final)

