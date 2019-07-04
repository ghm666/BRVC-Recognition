from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.layers import Convolution2D, MaxPooling2D
import keras
NUM_CLASSES = 2

def Lenet5(INPUT_SHAPE):

    model = Sequential()
    model.add(Convolution2D(3, kernel_size=(5, 5), padding='same', activation='relu', input_shape=INPUT_SHAPE))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Convolution2D(6, kernel_size=(5, 5), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Convolution2D(9, kernel_size=(5, 5), padding='same', activation='relu'))
    model.add(Flatten())  # 拉成一维
    model.add(Dense(500, activation='relu'))
    model.add(Dense(NUM_CLASSES, activation='softmax'))

    return model