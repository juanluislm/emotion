import pickle
import numpy as np
import cv2
from pandas import DataFrame
# from sklearn.metrics import confusion_matrix

import keras # high level api to tensorflow (or theano, CNTK, etc.) and useful image preprocessing
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from keras.models import Sequential, load_model, model_from_json
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
import glob
print('Keras image data format: {}'.format(K.image_data_format()))

def eval_model(model, test_data, gestures, base_path):

    labels = model.predict_generator(test_data, verbose=1)

    confusion_matrix = GetConfusion(test_data, labels, gestures)

    SaveToExcel(confusion_matrix, base_path)

def SaveToExcel(confusion, root):

    table_trans = confusion.transpose()

    df = DataFrame({'Label': gestures})

    for i in range(0, len(gestures)):
        df[gestures[i]] = table_trans[i]

    df.to_excel(root + '_confusion.xlsx', sheet_name='sheet1', index=False)
    print(df)

def GetConfusion(test_data, labels, gestures):

    confusion = np.zeros((len(gestures), len(gestures)), dtype=np.float64)
    label_count = np.zeros((len(gestures),))
    for i in range(0, len(labels)):
        class_idx = test_data.classes[i]
        predicted = np.argmax(labels[i])
        # print(predicted, class_idx, labels[i])

        #     print(class_idx, predicted)

        confusion[class_idx][predicted] += 1
        label_count[class_idx] += 1

    for i in range(0, len(gestures)):
        for j in range(0, len(gestures)):
            confusion[i][j] = confusion[i][j] / label_count[i]

    # predictions =  np.zeros( (len(labels),) )
    # for i in range(0, len(labels)):
    #     predictions[i] = np.argmax(labels[i])
    #
    # confusion = confusion_matrix(test_data.classes, predictions)

    return np.array(confusion)

def batch_test(files, gestures, generators):

    for file in files:
        print(file)
        test_data = None
        if("32x32x3" in file):
            test_data = generators[0]
        elif("64x64" in file):
            test_data = generators[2]
        else:
            test_data = generators[1]

        model = load_model(file, compile=False)

        eval_model(model, test_data, gestures, file)

def SetUpDataGenerators(val_path, w, h, rgb):
    batch_size = 16

    validation_datagen = ImageDataGenerator()#zoom_range=0.2, rotation_range=10)

    if(rgb == 3):
        validation_generator = validation_datagen.flow_from_directory(
            val_path,
            target_size=(w, h),
            # color_mode='grayscale',
            batch_size=batch_size
        )
    else:

        validation_generator = validation_datagen.flow_from_directory(
            val_path,
            target_size=(w, h),
            color_mode='grayscale',
            batch_size=batch_size
        )

    return validation_generator

if __name__ == '__main__':

    gestures = ['nohand','peace','stop','thumbsup']
    root = '/Users/jmarcano/dev/withme/emotion/keras/models/'

    files = glob.glob(root+'*/*.hdf5')

    data_root = '/Users/jmarcano/dev/withme/HandGesturesAndTracking/images/CommonHandGestures/validation_data'
    test = [SetUpDataGenerators(data_root, 32, 32, 3), SetUpDataGenerators(data_root, 32, 32, 1),
            SetUpDataGenerators(data_root, 64, 64, 1)]

    batch_test(files, gestures, test)