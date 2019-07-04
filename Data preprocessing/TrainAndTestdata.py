# 将训练图像和标签分别加载到一个数组

import os
import numpy as np

input_dir = "F:/BRVOdataset/new data/train"

def file2numpy(filename):
    fr = open(filename)
    count = len(fr.readlines())
    data_x = np.empty((count, 1, 60, 60), dtype="float32")
    fr = open(filename)
    data_y = []
    i = 0
    for line in fr.readlines():
        line = line.strip()
        listFromLine = line.split(' ')
        data_x[i, :, :, :] = np.load(input_dir+"/"+listFromLine[0])
        data_y.append(int(listFromLine[1]))
        i += 1
    return data_x,data_y

X_train,Y_train = file2numpy("F:/BRVOdataset/new data/train.txt")
# Y_test,Y_test = file2numpy("F:/BRVOdataset/new data/train.txt")


