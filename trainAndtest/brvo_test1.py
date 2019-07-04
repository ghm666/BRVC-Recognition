# BRVO模型单张图像测试
from keras.models import load_model
import numpy as np
import cv2

import numpy as np
from LenetCNN import Lenet5
from keras import backend as K
import keras
import matplotlib.pyplot as plt


KERNAL_SIZE = (5,5)
INPUT_SHAPE = (60,60,1)
NUM_CLASSES = 2
BATCH_SIZE = 64
EPOCHS = 40
img_rows, img_cols = 60,60



def precision(y_true, y_pred):
    """Precision metric. Only computes a batch-wise average of precision.
-    Computes the precision, a metric for multi-label classification of
-    how many selected items are relevant.
-    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def recall(y_true, y_pred):
    """Recall metric.
-    Only computes a batch-wise average of recall.
-    Computes the recall, a metric for multi-label classification of
-    how many relevant items are selected.
-    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def fbeta_score(y_true, y_pred, beta=1):

    """Computes the F score.

-    """
    if beta < 0:
        raise ValueError('The lowest choosable beta is zero (only precision).')

    # If there are no true positives, fix the F score at 0 like sklearn.
    if K.sum(K.round(K.clip(y_true, 0, 1))) == 0:

        return 0
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    bb = beta ** 2
    fbeta_score = (1 + bb) * (p * r) / (bb * p + r + K.epsilon())
    return fbeta_score

def fmeasure(y_true, y_pred):
    """Computes the f-measure, the harmonic mean of precision and recall.
    Here it is only computed as a batch-wise average, not globally.
    """
    return fbeta_score(y_true, y_pred, beta=1)

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

# img_path = "F:/BRVOdataset/BRVO_Rename/BRVO328.jpg"
img_path = "F:/BRVOdataset/Normal/normal94.jpg"
img = cv2.imread(img_path)
img = cv2.resize(img,(768,576))
cv2.imshow("orign", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

x = Imgperpross(img_path)

model = Lenet5(INPUT_SHAPE)
model.load_model("../model/mymodel40.h5")

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=[fmeasure,recall,precision])

y = model.predict(x)
print('Predicted:', y)




