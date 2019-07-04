from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing.image import ImageDataGenerator, array_to_img
from keras.models import Model, Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.optimizers import SGD
from keras import backend as K
import pandas as pd
import numpy as np

# dimensions of our images.
img_width, img_height = 224, 224

train_data_dir = 'F:/BRVOdataset/data/train'
validation_data_dir = 'F:/BRVOdataset/data/validation'
nb_train_samples = 390
nb_validation_samples = 52
epochs = 50
batch_size = 16

if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)


def preprocess_input_vgg(x):
    X = np.expand_dims(x, axis=0)
    X = preprocess_input(X)
    return X[0]

vgg16 = VGG16(weights='imagenet')

x = vgg16.get_layer('fc2').output
prediction = Dense(2, activation='softmax', name='predictions')(x)

model = Model(inputs=vgg16.input, outputs=prediction)


for layer in model.layers:
    if layer.name in ['predictions']:
        continue
    layer.trainable = False

df = pd.DataFrame(([layer.name, layer.trainable] for layer in model.layers), columns=['layer', 'trainable'])

train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input_vgg,
                                   rotation_range=40,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True,
                                   fill_mode='nearest')

train_generator = train_datagen.flow_from_directory(directory='F:/BRVOdataset/data/train',
                                                    target_size=[img_width, img_height],
                                                    batch_size=batch_size,
                                                    class_mode='categorical')

validation_datagen = ImageDataGenerator(preprocessing_function=preprocess_input_vgg,
                                        rotation_range=40,
                                        width_shift_range=0.2,
                                        height_shift_range=0.2,
                                        shear_range=0.2,
                                        zoom_range=0.2,
                                        horizontal_flip=True,
                                        fill_mode='nearest')

validation_generator = validation_datagen.flow_from_directory(directory='F:/BRVOdataset/data/validation',
                                                              target_size=[img_width, img_height],
                                                              batch_size=batch_size,
                                                              class_mode='categorical')

# Compile with SGD Optimizer and a Small Learning Rate
sgd = SGD(lr=1e-4, momentum=0.9)
model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
# model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

model.fit_generator(
    train_generator,
    steps_per_epoch= 25,
    epochs=20,
    validation_data=validation_generator,
    validation_steps=2)
# validation_steps=32) #,
# callbacks=callbacks_list)

# Save trained weight
# model.save_weights('../vgg16_tf_cat_dog_final_dense2.h5')
model.save('../model/20epoch_model.h5')

