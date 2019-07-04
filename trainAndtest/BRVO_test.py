# BRVO模型单张图像测试
from keras.models import load_model
import numpy as np
import cv2


# 测试预处理好的npy文件
# x = np.load("normal93.npy")
# x = x/255
# x = np.reshape(x,(1,60,60,1))


# 测试单张图像
def Imgperpross(img_path):
    img = cv2.imread(img_path)
    # img = img[28:1988, 509:2494]
    img = img[76:1474, 112:1498]
    g = np.zeros((img.shape[0], img.shape[1]), dtype=img.dtype)
    g[:, :] = img[:, :, 1]
    clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(8, 8))
    img = clahe.apply(g)
    img = cv2.resize(img,(60,60))
    img = np.reshape(img,(1,60,60,1))
    img = img/255
    return img

img_path = "F:/BRVOdataset/BRVO_Rename/BRVO331.jpg"
# img_path = "F:/BRVOdataset/Normal/normal97.jpg"
img = cv2.imread(img_path)
img = cv2.resize(img,(768,576))
cv2.imshow("orign", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

x = Imgperpross(img_path)

model = load_model('../model/lenet_40epoch_model.h5')
y = model.predict(x)
print('Predicted:', y)




