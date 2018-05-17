from statistics import mode
import cv2
import numpy as np
import matplotlib.pyplot as plt
import json
import time

from TFInference import TFInference

# draw a bar graph on the left with a string label, position as zero indexed int, val from 0-1
def draw_bar(image, val, label, position):
    cv2.putText(image, label, (10, position * 20 + 20), cv2.FONT_HERSHEY_PLAIN, 0.5, (0, 255, 0), 1);
    cv2.rectangle(image, (130, position * 20 + 10), (130 + int(val * 100), (position + 1) * 20 + 4), (255, 0, 0), -1)


enable_emotion_tracking = True
enable_refined_landmark_tracking = True
enable_mouth_tracking = True

face_detector = cv2.CascadeClassifier('models/haarcascade_frontalface_default.xml')
# loading landmark model
landmark_inference = TFInference(model_path = 'models/mtcnn_onet.h5.pb',
                                 input_name = 'import/input_1',
                                 output_names = ['import/output_node0',
                                                'import/output_node1',
                                                'import/output_node2'])

if enable_emotion_tracking:
    # loading emotion model
    emotion_inference = TFInference(model_path = 'models/emotion_mini_XCEPTION_64x64_0.66_7ms.hdf5.pb',
                                    input_name = 'import/input_1',
                                    output_names = ['import/output_node0', 'import/add_4/add'])
    # getting input model shapes for inference
    emotion_target_size = emotion_inference.input_shape[0:2]
    emotion_labels = {0:'angry',1:'disgust',2:'fear',3:'happy',4:'sad',5:'surprise',6:'neutral'}
    # SVR from emotion network
    with open('models/mouth_weights_robust_0.1.json') as fi:
        mouth_weights = json.load(fi)
        mouth_coef_top_down = mouth_weights['top_down_coef']
        mouth_intercept_top_down = mouth_weights['top_down_intercept']
        mouth_coef_left_right = mouth_weights['left_right_coef']
        mouth_intercept_left_right = mouth_weights['left_right_intercept']

if enable_mouth_tracking:
    # mouth openness model
    mouth_openness_inference = TFInference(model_path = 'models/faceoff_kao_onet_32_lm_12.70-0.46-0.14.hdf5.pb',
                                           input_name = 'import/input_1',
                                           output_names = ['import/output_node0'])
    mouth_target_size = mouth_openness_inference.input_shape[0:2]

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
    draw_bar_position = 0
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
    landmark_pts_raw = []
    for i in range(0, 5):
        landmark_pts_raw.append((landmark_pts[i]*w+x, landmark_pts[i+5]*h+y))
        cv2.circle(rgb_image, (int(landmark_pts_raw[i][0]), int(landmark_pts_raw[i][1])), 2, (0, 255, 0))
    face_coordinates = [x1, y1, x2-x1, y2-y1]

    # --------------------------------------------------
    # step 3: inference of emotion nets for emotion and mouth openness
    # --------------------------------------------------
    if enable_mouth_tracking:
        mouth_center_raw = ((landmark_pts_raw[3][0] + landmark_pts_raw[4][0])/2.0,
                            (landmark_pts_raw[3][1] + landmark_pts_raw[4][1])/2.0)
        mouth_half_size = int(max(x2-x1, y2-y1) / 64 * mouth_target_size[0] / 2)
        mouth_center_rounded = (int(mouth_center_raw[0]), int(mouth_center_raw[1]))
        gray_mouth_raw = cv2.resize(gray_image[(mouth_center_rounded[1]-mouth_half_size): (mouth_center_rounded[1]+mouth_half_size),
                                               (mouth_center_rounded[0]-mouth_half_size): (mouth_center_rounded[0]+mouth_half_size)],
                                    (mouth_target_size))
        gray_mouth_one = ((gray_mouth_raw / 255.0) - 0.5) * 2
        gray_mouth_one = np.expand_dims(gray_mouth_one, -1)
        gray_mouth_one = np.expand_dims(gray_mouth_one, 0)
        start = time.time()
        mouth_prediction = mouth_openness_inference.run(gray_mouth_one)
        print("forward pass of mouth deep nets took {} ms".format((time.time() - start)*1e3))

        left_right = np.array([mouth_prediction[0][0][0] - mouth_prediction[0][0][2], \
                               mouth_prediction[0][0][1] - mouth_prediction[0][0][3]])
        top_down = np.array([mouth_prediction[0][0][4] - mouth_prediction[0][0][6], \
                             mouth_prediction[0][0][5] - mouth_prediction[0][0][7]])
        openness = np.array([mouth_prediction[0][0][8] - mouth_prediction[0][0][10], \
                             mouth_prediction[0][0][9] - mouth_prediction[0][0][11]])
        left_right_length = np.linalg.norm(left_right)
        top_down_length = np.linalg.norm(top_down)
        openness_length = np.linalg.norm(openness)

        for i in range(6):
            cv2.circle(gray_mouth_raw, (int(mouth_prediction[0][0][i*2]), int(mouth_prediction[0][0][i*2+1])),
                       1, (255), -1)
            cv2.imshow('mouth input 0 {}'.format(mouth_target_size), gray_mouth_raw)

    if enable_emotion_tracking:
        gray_face = gray_image[y1:y2, x1:x2]
        gray_face = cv2.resize(gray_face, (emotion_target_size))
        gray_mouth = gray_face.copy()
        gray_mouth.fill(127.5)
        mouth_left_corner = ((landmark_pts[3]*w+x - x1) / (x2 - x1) * emotion_target_size[1],
                             (landmark_pts[8]*h+y - y1) / (y2 - y1) * emotion_target_size[0])
        mouth_right_corner = ((landmark_pts[4]*w+x - x1) / (x2 - x1) * emotion_target_size[1],
                              (landmark_pts[9]*h+y - y1) / (y2 - y1) * emotion_target_size[0])
        center_x = int((mouth_left_corner[0] + mouth_right_corner[0])/2)
        center_y = int((mouth_left_corner[1] + mouth_right_corner[1])/2)
        for i in range(max(0, center_x-16), min(center_x+16, emotion_target_size[0]-1)):
            for j in range(max(center_y-8, 0), min(center_y+8, emotion_target_size[1]-1)):
                gray_mouth[j - max(center_y-8, 0) + 40][i-max(0, center_x-16)+16] = gray_face[j][i]
        cv2.imshow('mouth input', gray_mouth)

        gray_face = ((gray_face / 255.0) - 0.5) * 2
        gray_face = np.expand_dims(gray_face, -1)
        gray_mouth = ((gray_mouth / 255.0) - 0.5) * 2
        gray_mouth = np.expand_dims(gray_mouth, -1)

        gray_face = np.array([gray_face, gray_mouth])
        start = time.time()
        emotion_prediction, intermediate_prediction = emotion_inference.run(gray_face)
        print("forward pass of emotion deep nets took {} ms".format((time.time() - start)*1e3))

        for i, emotion in enumerate(emotion_prediction[0]):
            draw_bar(rgb_image, emotion, emotion_labels[i], draw_bar_position)
            draw_bar_position += 1

        intermediate_val = intermediate_prediction[1].reshape(16, 128).reshape(2048)
        mouth_openness_left_right = max(0, np.dot(intermediate_val, mouth_coef_left_right) + mouth_intercept_left_right)
        mouth_openness_top_down = max(0, np.dot(intermediate_val, mouth_coef_top_down) + mouth_intercept_top_down)
    draw_bar_position += 1
    if enable_emotion_tracking:
        draw_bar(rgb_image, mouth_openness_left_right, 'mouth lr SVR', draw_bar_position)
        draw_bar_position += 1
    if enable_mouth_tracking:
        draw_bar(rgb_image, openness_length / left_right_length, 'mouth lr DNN', draw_bar_position)
        draw_bar_position += 1
    draw_bar_position += 1

    bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    cv2.imshow('window_frame', bgr_image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
