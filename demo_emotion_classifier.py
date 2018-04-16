from statistics import mode
import cv2
import numpy as np
from keras.models import load_model
from keras.preprocessing import image
import matplotlib.pyplot as plt
import time

def load_image(image_path, grayscale=False, target_size=None):
    pil_image = image.load_img(image_path, grayscale, target_size)
    return image.img_to_array(pil_image)

def load_detection_model(model_path):
    detection_model = cv2.CascadeClassifier(model_path)
    return detection_model

def detect_faces(detection_model, gray_image_array):
    return detection_model.detectMultiScale(gray_image_array, 1.3, 5)

def apply_offsets(face_coordinates, offsets):
    x, y, width, height = face_coordinates
    x_off, y_off = offsets
    return (x - x_off, x + width + x_off, y - y_off, y + height + y_off)

# parameters for loading data and images
detection_model_path = 'models/haarcascade_frontalface_default.xml'
emotion_model_path = 'models/emotion_mini_XCEPTION_64x64_0.66_7ms.hdf5'
emotion_model_path = 'models/emotion_mini_XCEPTION_48x48_0.63_5ms.hdf5'
emotion_labels = {0:'angry',1:'disgust',2:'fear',3:'happy',4:'sad',5:'surprise',6:'neutral'}

# hyper-parameters for bounding boxes shape
frame_window = 1
emotion_offsets = (20, 40)

# loading models
face_detection = load_detection_model(detection_model_path)
emotion_classifier = load_model(emotion_model_path, compile=False)

# getting input model shapes for inference
emotion_target_size = emotion_classifier.input_shape[1:3]

# starting lists for calculating modes
emotion_window = []

# starting video streaming
cv2.namedWindow('window_frame')
video_capture = cv2.VideoCapture(0)
while True:
    bgr_image = video_capture.read()[1]
    gray_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    faces = detect_faces(face_detection, gray_image)

    for face_coordinates in faces:
        x1, x2, y1, y2 = apply_offsets(face_coordinates, emotion_offsets)
        gray_face = gray_image[y1:y2, x1:x2]
        try:
            gray_face = cv2.resize(gray_face, (emotion_target_size))
        except:
            continue

        gray_face = ((gray_face / 255.0) - 0.5) * 2
        gray_face = np.expand_dims(gray_face, 0)
        gray_face = np.expand_dims(gray_face, -1)
        print(emotion_target_size)
        start = time.time()
        emotion_prediction = emotion_classifier.predict(gray_face)
        print("one frame took {} ms".format((time.time() - start)*1e3))

        for i, emotion in enumerate(emotion_prediction[0]):
            cv2.putText(rgb_image, emotion_labels[i], (10, i * 20 + 20), cv2.FONT_HERSHEY_PLAIN, 0.5, (0, 255, 0), 1);
            cv2.rectangle(rgb_image, (130, i * 20 + 10), (130 + int(emotion_prediction[0][i] * 100), (i + 1) * 20 + 4), (255, 0, 0), -1)

        x, y, w, h = face_coordinates
        cv2.rectangle(rgb_image, (x, y), (x + w, y + h), (255, 255, 255), 2)

    bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    cv2.imshow('window_frame', bgr_image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
