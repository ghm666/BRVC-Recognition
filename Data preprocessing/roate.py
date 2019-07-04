import cv2
import numpy as np
import os

input_dir = "F:/BRVOdataset/normal_green"
output_dir = "F:/BRVOdataset/normal_green_roate"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def rotate(image, angle, center=None, scale=1.0):
    # 获取图像尺寸
    (h, w) = image.shape[:2]

    # 若未指定旋转中心，则将图像中心设为旋转中心
    if center is None:
        center = (w / 2, h / 2)

    # 执行旋转
    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(image, M, (w, h))

    # 返回旋转后的图像
    return rotated

index = 1
for (path, dirnames, filenames) in os.walk(input_dir):
    for filename in filenames:
        if filename.endswith('.npy'):
            print('正在处理第 %s 张图片' % index)
            img_path = path + '/' + filename
            g = np.load(img_path)
            rotated = rotate(g, 90)
            newname = output_dir + "/normal_roate" + str(index) + ".npy"
            np.save(newname, rotated)
            index += 1