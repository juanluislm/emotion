from keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from CustomCallbacks import *
import pickle
from models import mini_XCEPTION, tiny_XCEPTION
from data import load_emotion_data, split_data
import datetime
import os
import copy
from GenericNetworkBuilder import *
import glob
import math
import random
import cv2
from sklearn.cross_validation import train_test_split


def SetUpDataGenerators(train_path, val_path, batch_size, w, h):
    # batch_size = 16

    training_datagen = ImageDataGenerator(featurewise_center=False,
                                          featurewise_std_normalization=False,
                                          rotation_range=10,
                                          width_shift_range=0.1,
                                          height_shift_range=0.1,
                                          zoom_range=.1,
                                          horizontal_flip=True)

    validation_datagen = ImageDataGenerator()

    training_generator = training_datagen.flow_from_directory(
        train_path,
        target_size=(w, h),
        # color_mode='grayscale',
        batch_size=batch_size

    )

    validation_generator = validation_datagen.flow_from_directory(
        val_path,
        target_size=(w, h),
        # color_mode='grayscale',
        batch_size=batch_size
    )

    return training_generator, validation_generator


def load_data(root, split, input_shape, classes):
    X_train = []
    y_train = []
    print('Read train images')

    #     classes = glob.glob(root+'*')
    #     print(classes)

    for j in range(0, len(classes)):
        print('Load folder {}'.format(j))

        files = glob.glob(root + classes[j] + '/*')
        print(len(files))

        for fl in files:
            img = get_im(fl, input_shape)
            X_train.append(img)
            y_train.append(j)

    X_train, X_val, y_train, y_val = split_validation_set(np.array(X_train), np.array(y_train), split)

    print("Using {} imgs for training and {} imgs for validation".format(len(X_train), len(X_val)))

    y_train2 = np.zeros((len(y_train), 4), dtype=np.float64)
    y_val2 = np.zeros((len(y_val), 4), dtype=np.float64)

    for i in range(0, len(y_train)):
        y_train2[i][y_train[i]] = 1

    for i in range(0, len(y_val)):
        y_val2[i][y_val[i]] = 1

    return X_train, y_train2, X_val, y_val2


def get_im(path, input_shape):
    # Load as grayscale
    # print(path)
    try:
        img = cv2.imread(path)

        # Reduce size
        resized = cv2.resize(img, (input_shape[0], input_shape[1]))
        if (input_shape[2] == 1):
            resized = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

        resized = ((resized / 255.0) - 0.5) * 2.0
    except:
        print("something is off with {}".format(path))
        return

    return resized


def split_validation_set(train, labels, test_size):
    random_state = 51
    X_train, X_test, y_train, y_test = train_test_split(train, labels, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test


# parameters
batch_size = 32  # * 4
num_epochs = 10000

conv2d_filters = [2, 4, 8, 12, 16, 32]
# input_shapes = [(32, 32, 3), (48, 48, 3), (64, 64, 3), (80, 80, 3)]
input_shape = (64, 64, 3)
residual_layers = range(1, 10)
conv_layers = range(1, 10)

validation_split = .2
verbose = 1
num_classes = 4
patience = 50
gestures = ['nohand', 'peace', 'stop', 'thumbsup']
root = '/Users/jmarcano/dev/withme/HandGesturesAndTracking/images/CommonHandGestures/training_data/'

root_test = '/Users/jmarcano/dev/withme/HandGesturesAndTracking/images/CommonHandGestures/testing_data/'
x_train, y_train, x_val, y_val = load_data(root_test, 0.2, input_shape, gestures )

# x_test, y_test, dummy1, dummy2 = load_data(root_test, 0.0, input_shape, gestures)

# y_train2 = np.zeros( (len(y_train), 4), dtype=np.float64)
# y_val2 = np.zeros( (len(y_val), 4), dtype=np.float64)

# for i in range(0, len(y_train)):
#     y_train2[i][ y_train[i]] =1
#
# for i in range(0, len(y_val)):
#     y_val2[i][ y_val[i] ] =1
# fdata = open('train_val_data_{}_{}_{}.pickle'.format(input_shape[0], input_shape[1], input_shape[2]), 'wb')

# pickle.dump([x_train, y_train, x_val, y_val], fdata)

# fdata.close()

# fdata2 = open('test_{}_{}_{}.pickle'.format(input_shape[0], input_shape[1], input_shape[2]), 'wb')


# pickle.dump([x_test, y_test], fdata2)

# fdata2.close()

conv_layer =3
residual_layer=3
conv2d_filter=8
print("a")
now = datetime.datetime.now()
base_path = 'arch_eval2/'
base_path += now.strftime("%Y_%m_%d_%H_%M_%S")+'/'
if not os.path.exists(base_path):
    os.makedirs(base_path)
# datasets_path = 'data/fer2013/fer2013.csv'
model_name = "ResidualNet_{}x{}x{}_conv_{}_res{}".format(input_shape[0], input_shape[1], input_shape[2],
                                                         6, 6)

model = ResidualNet(input_shape, num_classes, residual_layer, conv_layer, conv2d_filters=conv2d_filter)
model.compile(optimizer='adam', loss='categorical_crossentropy',
              metrics=['accuracy'])
model.summary()

# begin training
log_file_path = base_path + model_name+'_training.log'
csv_logger = CSVLogger(log_file_path, append=False)
early_stop = EarlyStopping('val_loss', patience=patience)
reduce_lr = ReduceLROnPlateau('val_loss', factor=0.1,
                              patience=int(patience/4), verbose=1)
trained_models_path = base_path + model_name
model_names = trained_models_path + '.{epoch:02d}-{val_acc:.2f}.hdf5'

history = model.fit(x=x_train,
                    y=y_train2,
                    epochs=2, verbose=2,
                    validation_data=(x_val, y_val2),
                    steps_per_epoch=len(x_train) // batch_size,
                    validation_steps=len(x_val) // batch_size
                    )

model.save("aaa.hdf5")