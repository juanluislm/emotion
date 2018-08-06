import sys  # system functions (ie. exiting the program)
import os  # operating system functions (ie. path building on Windows vs. MacOs)
import time  # for time operations
import uuid  # for generating unique file names
import math  # math functions
from keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping
from keras.callbacks import ReduceLROnPlateau
import datetime

# from IPython.display import display as ipydisplay, Image, clear_output, HTML  # for interacting with the notebook better

import numpy as np  # matrix operations (ie. difference between two matricies)
import cv2  # (OpenCV) computer vision functions (ie. tracking)

(major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')
print('OpenCV Version: {}.{}.{}'.format(major_ver, minor_ver, subminor_ver))

import matplotlib.pyplot as plt  # (optional) for plotting and showing images inline
# % matplotlib
# inline

import keras  # high level api to tensorflow (or theano, CNTK, etc.) and useful image preprocessing
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from keras.models import Sequential, load_model, model_from_json
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D

print('Keras image data format: {}'.format(K.image_data_format()))


def SetUpDataGenerators(train_path, val_path, batch_size, w, h):
    batch_size = 16

    # training_datagen = ImageDataGenerator(
    #     rotation_range=50,
    #     width_shift_range=0.1,
    #     height_shift_range=0.1,
    #     shear_range=0.2,
    #     zoom_range=0.2,
    #     horizontal_flip=True,
    #     fill_mode='nearest'
    #
    # )

    training_datagen = ImageDataGenerator(
        rotation_range=0,
        width_shift_range=0.0,
        height_shift_range=0.0,
        shear_range=0.0,
        zoom_range=0.0,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    validation_datagen = ImageDataGenerator()

    training_generator = training_datagen.flow_from_directory(
        train_path,
        target_size=(w, h),
        batch_size=batch_size,
        color_mode='grayscale'
    )

    validation_generator = validation_datagen.flow_from_directory(
        val_path,
        target_size=(w, h),
        batch_size=batch_size,
        color_mode='grayscale'
    )

    return training_generator, validation_generator


def SetupModel(w, h):
    model = Sequential()

    model.add(Conv2D(32, (3, 3), input_shape=(w, h, 1)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    return model

w =32
h =32
batch_size = 16
training_generator, validation_generator = SetUpDataGenerators('images/CommonHandGestures/training_data',
                                                               'images/CommonHandGestures/validation_data',
                                                                batch_size,
                                                                w,
                                                                h)

model = SetupModel(w, h)

now = datetime.datetime.now()
patience = 50
base_path = 'models/'
base_path += now.strftime("%Y_%m_%d_%H_%M_%S") + '/'

if not os.path.exists(base_path):
    os.makedirs(base_path)
# datasets_path = 'data/fer2013/fer2013.csv'
model_name = "mini_XCEPTION_{}x{}_".format(w, h)

log_file_path = base_path + 'gestures_training.log'
csv_logger = CSVLogger(log_file_path, append=False)
early_stop = EarlyStopping('val_loss', patience=patience)
reduce_lr = ReduceLROnPlateau('val_loss', factor=0.1,
                              patience=int(patience/4), verbose=1)
trained_models_path = base_path + 'gestures_' + model_name
model_names = trained_models_path + '.{epoch:02d}-{val_acc:.2f}.hdf5'
model_checkpoint = ModelCheckpoint(model_names, 'val_loss', verbose=1,
                                   save_best_only=True)
callbacks = [model_checkpoint, csv_logger, early_stop, reduce_lr]


history = model.fit_generator(
    generator=training_generator,
    steps_per_epoch=429,# // batch_size,
    epochs=1000,
    verbose=1, callbacks=callbacks,
    validation_data=validation_generator,
    validation_steps=100,# // batch_size,
    workers=8,
)