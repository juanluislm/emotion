import sys  # system functions (ie. exiting the program)
import os  # operating system functions (ie. path building on Windows vs. MacOs)
import time  # for time operations
import uuid  # for generating unique file names
import math  # math functions
from keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping
from keras.callbacks import ReduceLROnPlateau
import datetime
from CustomCallbacks import *

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
from keras import regularizers

from keras.layers import Convolution2D


#init the model
model= Sequential()

#add conv layers and pooling layers 
model.add(Convolution2D(32,3,3, input_shape=(64,64,1),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Convolution2D(32,3,3, input_shape=(64,64,1),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Dropout(0.5)) #to reduce overfitting

model.add(Flatten())

#Now two hidden(dense) layers:
model.add(Dense(output_dim = 150, activation = 'relu',
                kernel_regularizer=regularizers.l2(0.01)))

model.add(Dropout(0.5))#again for regularization

model.add(Dense(output_dim = 150, activation = 'relu',
                kernel_regularizer=regularizers.l2(0.01)))


model.add(Dropout(0.5))#last one lol

model.add(Dense(output_dim = 150, activation = 'relu',
                kernel_regularizer=regularizers.l2(0.01)))

#output layer
model.add(Dense(output_dim = 4, activation = 'sigmoid'))


#Now copile it
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


#Now generate training and test sets from folders

train_datagen=ImageDataGenerator(
                                   rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.,
                                   horizontal_flip = False
                                 )

test_datagen=ImageDataGenerator(rescale=1./255)

root = '/Users/jmarcano/dev/withme/HandGesturesAndTracking/images/CommonHandGestures'

training_set=train_datagen.flow_from_directory(root+"/training_data",
                                               target_size = (64,64),
                                               color_mode='grayscale',
                                               batch_size=32,
                                               class_mode='categorical')

test_set=test_datagen.flow_from_directory(root+"/validation_data",
                                               target_size = (64,64),
                                               color_mode='grayscale',
                                               batch_size=32,
                                               class_mode='categorical')


w=64
h=64
gestures = ['nohand','peace','stop','thumbsup']

now = datetime.datetime.now()
patience = 50
base_path = 'models/'
base_path += now.strftime("%Y_%m_%d_%H_%M_%S") + '/'

if not os.path.exists(base_path):
    os.makedirs(base_path)
# datasets_path = 'data/fer2013/fer2013.csv'
model_name = "handv2_{}x{}_".format(w, h)

log_file_path = base_path + 'gestures_training.log'
csv_logger = CSVLogger(log_file_path, append=False)
early_stop = EarlyStopping('val_loss', patience=patience)
reduce_lr = ReduceLROnPlateau('val_loss', factor=0.1,
                              patience=int(patience/4), verbose=1)
trained_models_path = base_path + 'gestures_' + model_name
model_names = trained_models_path + '.{epoch:02d}-{val_acc:.2f}.hdf5'

model_checkpoint = ModelCheckpointConfusion(model_names, 'val_loss', verbose=1,
                                            save_best_only=True, gestures=gestures, generator=test_set )

callbacks = [model_checkpoint, csv_logger, early_stop, reduce_lr]


#finally, start training
model.fit_generator(training_set,
                         steps_per_epoch = 60000 // 32,
                         epochs=10000,
                         validation_data = test_set,
                         nb_val_samples = 100,
                         verbose=1, callbacks=callbacks,
                         workers=8)


#after 10 epochs:
    #training accuracy: 0.9005
    #training loss:     0.4212
    #test set accuracy: 0.8813
    #test set loss:     0.5387

#saving the weights
# model.save("handv2.hdf5",overwrite=True)
#
# #saving the model itself in json format:
# model_json = model.to_json()
# with open("model.json", "w") as model_file:
#     model_file.write(model_json)
# print("Model has been saved.")
#
#
# #testing it to a random image from the test set
# img = load_img('Dataset/test_set/stop/stop26.jpg',target_size=(200,200))
# x=array(img)
# img = cv2.cvtColor( x, cv2.COLOR_RGB2GRAY )
# img=img.reshape((1,)+img.shape)
# img=img.reshape(img.shape+(1,))
#
# test_datagen = ImageDataGenerator(rescale=1./255)
# m=test_datagen.flow(img,batch_size=1)
# y_pred=model.predict_generator(m,1)
#
#
# #save the model schema in a pic
# plot_model(model, to_file='model.png', show_shapes = True)






