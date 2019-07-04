import cv2
import numpy as np
import os

input_dir = "F:/BRVOdataset/normal_green"
output_dir = "F:/BRVOdataset/normal_green_flip"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

index = 1
for (path, dirnames, filenames) in os.walk(input_dir):
    for filename in filenames:
        if filename.endswith('.npy'):
            print('正在处理第 %s 张图片' % index)
            img_path = path + '/' + filename
            g = np.load(img_path)
            flipped = cv2.flip(g, 1)
            newname = output_dir + "/normal_filp" + str(index) + ".npy"
            np.save(newname, flipped)
            index += 1