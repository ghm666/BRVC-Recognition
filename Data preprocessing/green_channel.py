# !/usr/bin/env python
# encoding: utf-8
# 提取图像绿色通道并保存为numpy数组并且进行局部直方图均衡化

import cv2
import numpy as np
import os

input_dir = "F:\\doctor_labeled_3class\\crvo1"
output_dir = "F:\\doctor_labeled_3class\\crvo2"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# index = 1
# for (path, dirnames, filenames) in os.walk(input_dir):
#     # imageNames.sort(key=lambda x:int(x[:-4]))
#     for filename in filenames:
#         # imageName.sort(key=lambda x:int(x[:-4]))
#         print(filename)
#         if filename.endswith('.jpg'):
#             print('正在处理第 %s 张图片' % index)
#             img_path = path + '\\' + filename
#             img = cv2.imread(img_path)
#             g = np.zeros((img.shape[0], img.shape[1]), dtype=img.dtype)
#             #提取绿色通道
#             g[:, :] = img[:, :, 1]
#             clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(8, 8))
#             Lhe = clahe.apply(g)
#             newname = output_dir + "\\normal" + str(index) + ".npy"
#             np.save(newname, Lhe)
#             index += 1


index = 1
imageNames = os.listdir(input_dir)
imageNames.sort(key=lambda x:int(x[4:-4]))

for filename in imageNames:

    print(filename)
    if filename.endswith('.jpg'):
        print('正在处理第 %s 张图片' % index)
        img_path = input_dir + '\\' + filename
        img = cv2.imread(img_path)
        g = np.zeros((img.shape[0], img.shape[1]), dtype=img.dtype)
        #提取绿色通道
        g[:, :] = img[:, :, 1]
        clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(8, 8))
        Lhe = clahe.apply(g)
        newname = output_dir + "\\crvo" + str(index) + ".npy"
        np.save(newname, Lhe)
        index += 1