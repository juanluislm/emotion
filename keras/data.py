import pandas as pd
import numpy as np
import cv2

def load_emotion_data(datasets_path, input_shape=(48, 48, 1)):
    # load data
    data = pd.read_csv(datasets_path)
    pixels = data['pixels'].tolist()
    width, height = 48, 48
    faces = []
    for pixel_sequence in pixels:
        face = [int(pixel) for pixel in pixel_sequence.split(' ')]
        face = np.asarray(face).reshape(width, height)
        face = cv2.resize(face.astype('uint8'), input_shape[:2])
        faces.append(face.astype('float32'))
    faces = np.asarray(faces)
    faces = np.expand_dims(faces, -1)
    faces = faces / 255.0
    faces = faces - 0.5
    faces = faces * 2.0
    emotions = pd.get_dummies(data['emotion']).as_matrix()
    return faces, emotions

def split_data(x, y, split=.2):
    num_samples = len(x)
    num_train_samples = int((1 - split)*num_samples)
    train_x = x[:num_train_samples]
    train_y = y[:num_train_samples]
    val_x = x[num_train_samples:]
    val_y = y[num_train_samples:]
    train_data = (train_x, train_y)
    val_data = (val_x, val_y)
    return train_data, val_data
