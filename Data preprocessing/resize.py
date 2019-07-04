# -*- coding=utf-8 -*-
import os
import cv2

# 输入图片的
input_dir = "F:\\doctor_labeled_3class\\resize\\normal"
output_dir = "F:\\doctor_labeled_3class\\resize\\normal1"

width = 224
height = 224

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

index = 1
for (path, dirnames, filenames) in os.walk(input_dir):
    for filename in filenames:
        if filename.endswith('.jpg'):
            print('正在处理第 %s 张图片' % index)
            img_path = path + '/' + filename
            img = cv2.imread(img_path)
            new_img = cv2.resize(img, (width, height),interpolation = cv2.INTER_AREA)
            cv2.imwrite(output_dir + '/' + filename, new_img)
            index += 1