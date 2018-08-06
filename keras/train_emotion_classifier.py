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
input_shape = (64, 64, 3)
validation_split = .2
verbose = 1
num_classes = 4
patience = 50
root = '/Users/jmarcano/dev/withme/HandGesturesAndTracking/images/CommonHandGestures'
training_generator, validation_generator = SetUpDataGenerators(root+'/training_data',
                                                               root+'/validation_data',
                                                                batch_size,
                                                                input_shape[0],
                                                                input_shape[1])

gestures = ['nohand','peace','stop','thumbsup']
now = datetime.datetime.now()
base_path = 'models/'
base_path += now.strftime("%Y_%m_%d_%H_%M_%S")+'/'
if not os.path.exists(base_path):
    os.makedirs(base_path)
# datasets_path = 'data/fer2013/fer2013.csv'
model_name = "mini_XCEPTION_{}x{}x{}_".format(input_shape[0], input_shape[1], input_shape[2])


# model parameters/compilation
model = mini_XCEPTION(input_shape, num_classes)
model.compile(optimizer='adam', loss='categorical_crossentropy',
              metrics=['accuracy'])
model.summary()

# begin training
log_file_path = base_path + 'emotion_training.log'
csv_logger = CSVLogger(log_file_path, append=False)
early_stop = EarlyStopping('val_loss', patience=patience)
reduce_lr = ReduceLROnPlateau('val_loss', factor=0.1,
                              patience=int(patience/4), verbose=1)
trained_models_path = base_path + 'emotion_' + model_name
model_names = trained_models_path + '.{epoch:02d}-{val_acc:.2f}.hdf5'
# model_checkpoint = ModelCheckpoint(model_names, 'val_loss', verbose=1, save_best_only=True)

model_checkpoint = ModelCheckpointConfusion(model_names, 'val_loss', verbose=1,
                                            save_best_only=True, gestures=gestures, generator=validation_generator )

callbacks = [model_checkpoint, csv_logger, early_stop, reduce_lr]

# train_test = open(base_path+'train_test_data.pickle','wb')
# # pickle.dump([copy.deepcopy(training_generator), copy.deepcopy(validation_generator)], train_test)
# pickle.dump([training_generator, validation_generator], train_test)
# train_test.close()



history = model.fit_generator(
    generator=training_generator,
    steps_per_epoch= 60000 //batch_size,# // batch_size,
    epochs=num_epochs,
    verbose=1,callbacks=callbacks,
    validation_data=validation_generator,
    validation_steps=100,# // batch_size,
    workers=8,
)
#
# # model.save("models/emotion_model_gray.hdf5")
# history_file = open(base_path+'model_history.pickle','wb')
# pickle.dump(history.history, history_file)
# history_file.close()