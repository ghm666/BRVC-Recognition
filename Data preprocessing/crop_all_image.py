# -*- coding: UTF-8 -*-
# 对原始BRVO数据集进行裁剪,保留眼底图像中心区域
import os
import cv2

# 原始图片文件夹
imageDir = "F:\\brvo_crvo_nomal_dataset\\doctor_label_data\\newdata\\normal1"
# 裁剪之后文件夹
newimageDir ="F:\\brvo_crvo_nomal_dataset\\doctor_label_data\\newdata\\normal2"

imageName = os.listdir(imageDir)
index = 1
for imagename in imageName:
     #先输入Y的坐标，在输入X的坐标

     print(imagename)
     print('正在处理第 %s 张图片' % index)
     newDir = imageDir+"\\"+imagename
     image = cv2.imread(newDir)
     cropImg = image[45:1980, 523:2472]
     # cropImg = image[33:1968, 491:2450]
     oriname = newimageDir + "\\" + imagename
     cv2.imwrite(oriname, cropImg)
     index += 1
