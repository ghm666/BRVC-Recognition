import cv2
import numpy as np
import os
from numpy import *

def SaltAndPepper(src,percetage):
    SP_NoiseImg=src
    SP_NoiseNum=int(percetage*src.shape[0]*src.shape[1])
    for i in range(SP_NoiseNum):
        randX=random.random_integers(0,src.shape[0]-1)
        randY=random.random_integers(0,src.shape[1]-1)
        if random.random_integers(0,1)==0:
            SP_NoiseImg[randX,randY]=0
        else:
            SP_NoiseImg[randX,randY]=255
    return SP_NoiseImg


input_dir = "F:/BRVOdataset/BRVO_green"
output_dir = "F:/BRVOdataset/BRVO_green_noise"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

index = 1
for (path, dirnames, filenames) in os.walk(input_dir):
    for filename in filenames:
        if filename.endswith('.npy'):
            print('正在处理第 %s 张图片' % index)
            img_path = path + '/' + filename
            g = np.load(img_path)
            SaltAndPepper_noiseImage = SaltAndPepper(g, 0.05)
            newname = output_dir + "/BRVO_noise" + str(index) + ".npy"
            np.save(newname, SaltAndPepper_noiseImage)
            index += 1