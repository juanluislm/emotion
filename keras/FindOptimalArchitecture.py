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


def SetUpDataGenerators(train_path, val_path, batch_size, w, h):
    # batch_size = 16

    training_datagen =ImageDataGenerator(featurewise_center=False,
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

# parameters
batch_size = 32# * 4
num_epochs = 10000

conv2d_filters = [2, 4, 8, 12, 16, 32]
input_shapes = [(32, 32, 3), (48, 48, 3), (64, 64, 3), (80, 80, 3)]
residual_layers = range(1,10)
conv_layers = range(1,10)

validation_split = .2
verbose = 1
num_classes = 4
patience = 50
gestures = ['nohand','peace','stop','thumbsup']

for input_shape in input_shapes:

    root = '/Users/jmarcano/dev/withme/HandGesturesAndTracking/images/CommonHandGestures'
    training_generator, validation_generator = SetUpDataGenerators(root+'/training_data',
                                                               root+'/validation_data',
                                                                batch_size,
                                                                input_shape[0],
                                                                input_shape[1])

    for conv_layer in conv_layers:

        for residual_layer in residual_layers:

            for conv2d_filter in conv2d_filters:

                now = datetime.datetime.now()
                base_path = 'arch_eval/'
                base_path += now.strftime("%Y_%m_%d_%H_%M_%S")+'/'
                if not os.path.exists(base_path):
                    os.makedirs(base_path)
                # datasets_path = 'data/fer2013/fer2013.csv'
                model_name = "ResidualNet_{}x{}x{}_conv_{}_res{}".format(input_shape[0], input_shape[1], input_shape[2],
                                                                         conv_layer, residual_layer)

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
                # model_checkpoint = ModelCheckpoint(model_names, 'val_loss', verbose=1, save_best_only=True)

                model_checkpoint = ModelCheckpointConfusion(model_names, 'val_acc', verbose=1,
                                                save_best_only=True, gestures=gestures, generator=validation_generator )

                callbacks = [model_checkpoint, csv_logger, early_stop, reduce_lr]

                try:
                    history = model.fit_generator(
                        generator=training_generator,
                        steps_per_epoch=60000 // batch_size,  # // batch_size,
                        epochs=num_epochs,
                        verbose=1, callbacks=callbacks,
                        validation_data=validation_generator,
                        validation_steps=100,  # // batch_size,
                        workers=8,
                    )

                    pickle.dump(history, open(base_path + model_name + '_history.pickle', 'wb'))

                except:

                    print("-"*80)
                    print("Failed with params: input_shape {} , conv layers {} , residual layers {} , and conv filters {}".format(
                        input_shape, conv_layer, residual_layer, conv2d_filters
                    ))