from statistics import mode
import cv2
import numpy as np
import matplotlib.pyplot as plt
import json
import time

from TFInference import TFInference

# hyper-parameters for bounding boxes shape
emotion_offsets = (50, 50)

def apply_offsets(face_coordinates, offsets):
    x, y, width, height = face_coordinates
    x_off, y_off = offsets
    return (x - x_off, x + width + x_off, y - y_off, y + height + y_off)

with open('mouth_weights_absolute.json') as fi:
    mouth_weights = json.load(fi)
    mouth_coef_absolute = mouth_weights['coef']
    mouth_intercept_absolute = mouth_weights['intercept']

with open('mouth_weights_robust_0.1.json') as fi:
    mouth_weights = json.load(fi)
    mouth_coef_top_down = mouth_weights['top_down_coef']
    mouth_intercept_top_down = mouth_weights['top_down_intercept']
    mouth_coef_left_right = mouth_weights['left_right_coef']
    mouth_intercept_left_right = mouth_weights['left_right_intercept']

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
face_detector = cv2.CascadeClassifier('models/haarcascade_frontalface_default.xml')
# loading emotion model
emotion_inference = TFInference(model_path = 'models/emotion_mini_XCEPTION_64x64_0.66_7ms.hdf5.pb',
                                input_name = 'import/input_1',
                                output_names = ['import/output_node0', 'import/add_4/add'])
# getting input model shapes for inference
emotion_target_size = emotion_inference.input_shape[0:2]
emotion_labels = {0:'angry',1:'disgust',2:'fear',3:'happy',4:'sad',5:'surprise',6:'neutral'}

# loading landmark model
landmark_inference = TFInference(model_path = 'models/mtcnn_onet.h5.pb',
                                 input_name = 'import/input_1',
                                 output_names = ['import/output_node0',
                                                 'import/output_node1',
                                                 'import/output_node2'])

# starting video streaming
cv2.namedWindow('window_frame')
video_capture = cv2.VideoCapture(0)
face_tracking = False

while True:
    bgr_image = video_capture.read()[1]
    gray_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    height = bgr_image.shape[0]
    width = bgr_image.shape[1]
    # --------------------------------------------------
    # step 1: if face is not tracked, look for one face
    # --------------------------------------------------
    if not face_tracking:
        faces = face_detector.detectMultiScale(gray_image, 1.3, 5)
        if len(faces) > 0:
            face_coordinates = faces[0]
            face_tracking = True
        else:
            continue
    # --------------------------------------------------
    # step 2: inference of landmark nets for landmarks and face probability
    # --------------------------------------------------
    x, y, w, h = face_coordinates
    face_crop_original = bgr_image[y:(y+h), x:(x+w)]
    face_crop_original = cv2.resize(face_crop_original, (48, 48))
    face_crop = (face_crop_original - 127.5) / 127.5
    face_crop = np.expand_dims(face_crop, 0)

    start = time.time()
    landmark_prediction = landmark_inference.run(face_crop)
    print("forward pass of landmark deep nets took {} ms".format((time.time()-start)*1e3))
    landmark_prob = landmark_prediction[0][0][1]
    landmark_roi = landmark_prediction[1][0]
    landmark_pts = landmark_prediction[2][0]
    if landmark_prob < 0.7:
        face_tracking = False
        cv2.imshow('failed', face_crop_original)
        continue
    x1, y1, x2, y2 = [int(landmark_roi[0]*w+x), int(landmark_roi[1]*h+y),
                      int(landmark_roi[2]*w+x+w), int(landmark_roi[3]*h+y+h)]
    cv2.rectangle(rgb_image, (x, y), (x + w, y + h), (255, 255, 255), 2)
    cv2.rectangle(rgb_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    if (x2 - x1) > (y2 - y1):
        diff = (x2-x1)-(y2-y1)
        y1 = max(int(y1-diff/2), 0)
        y2 = min(int(y2+diff/2), height-1)
    else:
        diff = (y2-y1)-(x2-x1)
        x1 = max(int(x1-diff/2), 0)
        x2 = min(int(x2+diff/2), width-1)
    for i in range(0, 5):
        cv2.circle(rgb_image, (int(landmark_pts[i]*w+x), int(landmark_pts[i+5]*h+y)), 2, (0, 255, 0))

    # --------------------------------------------------
    # step 3: inference of emotion nets for emotion and mouth openness
    # --------------------------------------------------
    face_coordinates = [x1, y1, x2-x1, y2-y1]
    gray_face = gray_image[y1:y2, x1:x2]

    if True:
        gray_face = cv2.resize(gray_face, (emotion_target_size))
        gray_mouth = gray_face.copy()
        gray_mouth.fill(127.5)

        mouth_left_corner = ((landmark_pts[3]*w+x - x1) / (x2 - x1) * emotion_target_size[1],
                             (landmark_pts[8]*h+y - y1) / (y2 - y1) * emotion_target_size[0])
        mouth_right_corner = ((landmark_pts[4]*w+x - x1) / (x2 - x1) * emotion_target_size[1],
                              (landmark_pts[9]*h+y - y1) / (y2 - y1) * emotion_target_size[0])
        center_x = int((mouth_left_corner[0] + mouth_right_corner[0])/2)
        center_y = int((mouth_left_corner[1] + mouth_right_corner[1])/2)

        max_val = 0
        min_val = 255

        for i in range(max(0, center_x-16), min(center_x+16, emotion_target_size[0]-1)):
            for j in range(max(center_y-8, 0), min(center_y+8, emotion_target_size[1]-1)):
                gray_mouth[j - max(center_y-8, 0) + 40][i-max(0, center_x-16)+16] = gray_face[j][i]


        cv2.imshow('mouth input', gray_mouth)
        # gray_mouth = clahe.apply(gray_mouth)

    gray_face = ((gray_face / 255.0) - 0.5) * 2
    gray_face = np.expand_dims(gray_face, -1)

    gray_mouth = ((gray_mouth / 255.0) - 0.5) * 2
    gray_mouth = np.expand_dims(gray_mouth, -1)

    gray_face = np.array([gray_face, gray_mouth])
    start = time.time()
    emotion_prediction, intermediate_prediction = emotion_inference.run(gray_face)
    print("forward pass of emotion deep nets took {} ms".format((time.time() - start)*1e3))

    for i, emotion in enumerate(emotion_prediction[0]):
        cv2.putText(rgb_image, emotion_labels[i], (10, i * 20 + 20), cv2.FONT_HERSHEY_PLAIN, 0.5, (0, 255, 0), 1);
        cv2.rectangle(rgb_image, (130, i * 20 + 10), (130 + int(emotion * 100), (i + 1) * 20 + 4), (255, 0, 0), -1)

    intermediate_val = intermediate_prediction[1].reshape(16, 128).reshape(2048)
    mouth_openness_left_right = max(0, np.dot(intermediate_val, mouth_coef_left_right) + mouth_intercept_left_right)
    mouth_openness_top_down = max(0, np.dot(intermediate_val, mouth_coef_top_down) + mouth_intercept_top_down)
    mouth_openness_absolute = max(0, np.dot(intermediate_val, mouth_coef_absolute) + mouth_intercept_absolute)
    print('mouth ab:', mouth_openness_absolute)
    print('mouth lr:', mouth_openness_left_right)
    print('mouth td:', mouth_openness_top_down)

    cv2.putText(rgb_image, 'mouth absolute', (10, 8 * 20 + 20), cv2.FONT_HERSHEY_PLAIN, 0.5, (0, 255, 0), 1);
    cv2.rectangle(rgb_image, (130, 8 * 20 + 10), (130 + int(mouth_openness_absolute * 100), (8 + 1) * 20 + 4), (255, 0, 0), -1)
    cv2.putText(rgb_image, 'mouth left right', (10, 9 * 20 + 20), cv2.FONT_HERSHEY_PLAIN, 0.5, (0, 255, 0), 1);
    cv2.rectangle(rgb_image, (130, 9 * 20 + 10), (130 + int(mouth_openness_left_right * 100), (9 + 1) * 20 + 4), (255, 0, 0), -1)
    cv2.putText(rgb_image, 'mouth top down', (10, 10 * 20 + 20), cv2.FONT_HERSHEY_PLAIN, 0.5, (0, 255, 0), 1);
    cv2.rectangle(rgb_image, (130, 10 * 20 + 10), (130 + int(mouth_openness_top_down * 100), (10 + 1) * 20 + 4), (255, 0, 0), -1)

    bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    cv2.imshow('window_frame', bgr_image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
