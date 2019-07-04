import numpy as np
from LenetCNN import Lenet5
from keras import backend as K
import keras

INPUT_SHAPE = (60,60,1)
NUM_CLASSES = 2
BATCH_SIZE = 64
EPOCHS = 40
img_rows, img_cols = 60,60

train_dir = "F:/doctor_labeled_3class/data/train"
validation_dir = "F:/doctor_labeled_3class/data/validation"


def file2numpy(filename,dir):
    fr = open(filename)
    count = len(fr.readlines())
    data_x = np.empty((count, 1, 60, 60), dtype="float32")
    fr = open(filename)
    data_y = []
    i = 0
    for line in fr.readlines():
        line = line.strip()
        listFromLine = line.split(' ')
        data_x[i, :, :, :] = np.load(dir+"/"+listFromLine[0])
        data_y.append(int(listFromLine[1]))
        i += 1
    return data_x,data_y


x_train, y_train = file2numpy("F:/BRVOdataset/new data/train.txt",train_dir)
x_val, y_val = file2numpy("F:/BRVOdataset/new data/val.txt",validation_dir)


if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_val = x_val.reshape(x_val.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_val = x_val.reshape(x_val.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_val = x_val.astype('float32')
x_train /= 255
x_val /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_val.shape[0], 'val samples')

y_train = keras.utils.to_categorical(y_train, NUM_CLASSES)
y_val = keras.utils.to_categorical(y_val, NUM_CLASSES)
print('labels, categorical: ',y_train.shape, y_val.shape)


model = Lenet5(INPUT_SHAPE)
model.summary()

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

model.fit(x_train, y_train,epochs = EPOCHS,
                  batch_size=BATCH_SIZE,)



score = model.evaluate(x_val, y_val, verbose=0)
print('val loss:', score[0])
print('val accuracy:', score[1])