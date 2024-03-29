import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras import backend as K
import keras
import matplotlib.pyplot as plt


KERNAL_SIZE = (5,5)
INPUT_SHAPE = (60,60,1)
POOL_SIZE = (2,2)
NUM_CLASSES = 2
BATCH_SIZE = 64
EPOCHS = 40

img_rows, img_cols = 60,60

# the data, split between train and test sets
train_dir = "F:/BRVOdataset/new data/train"
test_dir = "F:/BRVOdataset/new data/test"


def file2numpy1(filename):
    fr = open(filename)
    count = len(fr.readlines())
    data_x = np.empty((count, 1, 60, 60), dtype="float32")
    fr = open(filename)
    data_y = []
    i = 0
    for line in fr.readlines():
        line = line.strip()
        listFromLine = line.split(' ')
        data_x[i, :, :, :] = np.load(train_dir+"/"+listFromLine[0])
        data_y.append(int(listFromLine[1]))
        i += 1
    return data_x,data_y

def file2numpy2(filename):
    fr = open(filename)
    count = len(fr.readlines())
    data_x = np.empty((count, 1, 60, 60), dtype="float32")
    fr = open(filename)
    data_y = []
    i = 0
    for line in fr.readlines():
        line = line.strip()
        listFromLine = line.split(' ')
        data_x[i, :, :, :] = np.load(test_dir+"/"+listFromLine[0])
        data_y.append(int(listFromLine[1]))
        i += 1
    return data_x,data_y


x_train, y_train = file2numpy1("F:/BRVOdataset/new data/train.txt")
x_test, y_test = file2numpy2("F:/BRVOdataset/new data/test.txt")


if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

y_train = keras.utils.to_categorical(y_train, NUM_CLASSES)
y_test = keras.utils.to_categorical(y_test, NUM_CLASSES)
print('labels, categorical: ',y_train.shape, y_test.shape)


model = Sequential()
model.add(Convolution2D(3,kernel_size=(5,5),padding='same',activation='relu',input_shape=INPUT_SHAPE))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Convolution2D(6,kernel_size=(5,5),padding='same',activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Convolution2D(9,kernel_size=(5,5),padding='same',activation='relu'))
model.add(Flatten()) #拉成一维
model.add(Dense(500, activation='relu'))
model.add(Dense(NUM_CLASSES, activation='softmax'))

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


#Woptimizer=keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08),
#sgd = keras.optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=False)
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=[fmeasure,recall,precision])

history = model.fit(x_train, y_train,epochs = EPOCHS,
                  batch_size=BATCH_SIZE,)

history_dict = history.history
history_dict.keys()

epochs = range(1, len(history_dict['loss']) + 1)


plt.plot(epochs, history_dict['loss'], 'b',label='loss')
plt.plot(epochs, history_dict['precision'], 'g',label='precision')

plt.xlabel('Epochs')
plt.grid()
plt.legend(loc=1)
plt.show()
model.save('../model/mymodel.h5')
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])