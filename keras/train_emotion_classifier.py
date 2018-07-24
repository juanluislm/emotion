from keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
import pickle
from models import mini_XCEPTION, tiny_XCEPTION
from data import load_emotion_data, split_data
import datetime
import os


def SetUpDataGenerators(train_path, val_path, batch_size, w, h):
    batch_size = 16

    training_datagen =ImageDataGenerator(featurewise_center=False,
                                    featurewise_std_normalization=False,
                                    rotation_range=10,
                                    width_shift_range=0.1,
                                    height_shift_range=0.1,
                                    zoom_range=.1,
                                    horizontal_flip=True)

    validation_datagen = ImageDataGenerator(zoom_range=0.2, rotation_range=10)

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

# parameters
batch_size = 32
num_epochs = 10000
input_shape = (64, 64, 1)
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

now = datetime.datetime.now()
base_path = 'models/'
base_path += now.strftime("%Y_%m_%d_%H_%M_%S") + '/'
if not os.path.exists(base_path):
    os.makedirs(base_path)
# datasets_path = 'data/fer2013/fer2013.csv'
model_name = "mini_XCEPTION_{}x{}_".format(input_shape[0], input_shape[1])

# load fer2013 dataset

# data generator
data_generator = ImageDataGenerator(featurewise_center=False,
                                    featurewise_std_normalization=False,
                                    rotation_range=10,
                                    width_shift_range=0.1,
                                    height_shift_range=0.1,
                                    zoom_range=.1,
                                    horizontal_flip=True)

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
model_checkpoint = ModelCheckpoint(model_names, 'val_loss', verbose=1,
                                   save_best_only=True)
callbacks = [model_checkpoint, csv_logger, early_stop, reduce_lr]

# loading dataset
# num_samples, num_classes = emotions.shape
# train_data, val_data = split_data(faces, emotions, validation_split)
# train_faces, train_emotions = train_data
# model.fit_generator(data_generator.flow(train_faces, train_emotions,
#                                         batch_size),
#                     steps_per_epoch=len(train_faces) / batch_size,
#                     epochs=num_epochs, verbose=1, callbacks=callbacks,
#                     validation_data=val_data)

history = model.fit_generator(
    generator=training_generator,
    steps_per_epoch= 55000 //32,# // batch_size,
    epochs=num_epochs,
    verbose=1,callbacks=callbacks,
    validation_data=validation_generator,
    validation_steps=100,# // batch_size,
    workers=8,
)

model.save("model/hand_model_gray.hdf5")
history_file = open('model/model_history.pickle','wb')
pickle.dump(history.history, history_file)
history_file.close()