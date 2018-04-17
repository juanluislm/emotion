from statistics import mode
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import time

def apply_offsets(face_coordinates, offsets):
    x, y, width, height = face_coordinates
    x_off, y_off = offsets
    return (x - x_off, x + width + x_off, y - y_off, y + height + y_off)

# parameters for loading data and images
detection_model_path = 'models/haarcascade_frontalface_default.xml'
# emotion_model_path = 'models/emotion_mini_XCEPTION_64x64_0.66_7ms.hdf5.pb'
emotion_model_path = 'models/emotion_mini_XCEPTION_48x48_0.63_5ms.hdf5.pb'
emotion_labels = {0:'angry',1:'disgust',2:'fear',3:'happy',4:'sad',5:'surprise',6:'neutral'}

# hyper-parameters for bounding boxes shape
frame_window = 1
emotion_offsets = (20, 40)

# loading models
face_detector = cv2.CascadeClassifier(detection_model_path)

# load tensorflow model
graph = tf.Graph()
graph_def = tf.GraphDef()
with open(emotion_model_path, "rb") as f:
    graph_def.ParseFromString(f.read())
with graph.as_default():
    tf.import_graph_def(graph_def)
input_name = 'import/input_1'
output_name = 'import/output_node0'
input_operation = graph.get_operation_by_name(input_name)
output_operation = graph.get_operation_by_name(output_name)
input_shape = (int(input_operation.outputs[0].shape.dims[1]),
               int(input_operation.outputs[0].shape.dims[2]),
               1)

# getting input model shapes for inference
emotion_target_size = input_shape[0:2]

# starting lists for calculating modes
emotion_window = []

# starting video streaming
cv2.namedWindow('window_frame')
video_capture = cv2.VideoCapture(0)
sess = tf.Session(graph = graph)
while True:
    bgr_image = video_capture.read()[1]
    gray_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    faces = face_detector.detectMultiScale(gray_image, 1.3, 5)

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
        emotion_prediction = sess.run(output_operation.outputs[0], {
            input_operation.outputs[0]: gray_face
        })
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
