# coding: utf-8

import numpy as np
import service
import cv2
import sys
import socketio

from threading import Thread
from queue import Queue, Full, Empty

source = int(sys.argv[1]) if sys.argv[1].isdigit() else sys.argv[1]
cap = cv2.VideoCapture(source)
# cap = cv2.VideoCapture(0)

fd = service.UltraLightFaceDetecion("weights/RFB-320.tflite",
                                    conf_threshold=0.98)
fa = service.CoordinateAlignmentModel("weights/coor_2d106.tflite")
hp = service.HeadPoseEstimator("weights/head_pose_object_points.npy",
                               cap.get(3), cap.get(4))
gs = service.IrisLocalizationModel("weights/iris_localization.tflite")

QUEUE_BUFFER_SIZE = 18

box_queue = Queue(maxsize=QUEUE_BUFFER_SIZE)
landmark_queue = Queue(maxsize=QUEUE_BUFFER_SIZE)
iris_queue = Queue(maxsize=QUEUE_BUFFER_SIZE)
upstream_queue = Queue(maxsize=QUEUE_BUFFER_SIZE)

# ======================================================

def put_nowait_drop_oldest(q, item):
    try:
        q.put_nowait(item)
    except Full:
        try:
            q.get_nowait()
        except Empty:
            pass
        q.put_nowait(item)

def face_detection():
    while True:
        ret, frame = cap.read()
        #frame = cv2.resize(frame, (500,300), interpolation=cv2.INTER_AREA)

        if not ret:
            break

        face_boxes, _ = fd.inference(frame)
        put_nowait_drop_oldest(box_queue, (frame, face_boxes))


def face_alignment():
    while True:
        frame, boxes = box_queue.get()
        landmarks = fa.get_landmarks(frame, boxes)
        put_nowait_drop_oldest(landmark_queue, (frame, landmarks))


def iris_localization(YAW_THD=45):
    sio = socketio.Client()

    sio.connect("http://127.0.0.1:6789", namespaces='/model')
    
    EYE_SMOOTH = 0.4
    left_eye_smooth = 1.0
    right_eye_smooth = 1.0

    while True:
        frame, preds = landmark_queue.get()

        for landmarks in preds:
            # calculate head pose
            euler_angle = hp.get_head_pose(landmarks).flatten()
            pitch, yaw, roll = euler_angle

            eye_starts = landmarks[[35, 89]]
            eye_ends = landmarks[[39, 93]]
            eye_centers = landmarks[[34, 88]]
            eye_lengths = (eye_ends - eye_starts)[:, 0]

            pupils = eye_centers.copy()

            if yaw > -YAW_THD:
                iris_left = gs.get_mesh(frame, eye_lengths[0], eye_centers[0])
                pupils[0] = iris_left[0]

            if yaw < YAW_THD:
                iris_right = gs.get_mesh(frame, eye_lengths[1], eye_centers[1])
                pupils[1] = iris_right[0]

            poi = eye_starts, eye_ends, pupils, eye_centers

            theta, pha, _ = gs.calculate_3d_gaze(poi)
            mouth_open_percent = (
                landmarks[60, 1] - landmarks[62, 1]) / (landmarks[53, 1] - landmarks[71, 1])
            left_eye_status = (
                landmarks[33, 1] - landmarks[40, 1]) / eye_lengths[0]
            right_eye_status = (
                landmarks[87, 1] - landmarks[94, 1]) / eye_lengths[1]
                
            left_eye_smooth = left_eye_smooth * (1 - EYE_SMOOTH) + left_eye_status * EYE_SMOOTH
            right_eye_smooth = right_eye_smooth * (1 - EYE_SMOOTH) + right_eye_status * EYE_SMOOTH

            result_string = {'euler': (pitch, -yaw, -roll),
                             'eye': (theta.mean(), pha.mean()),
                             'mouth': mouth_open_percent,
                             'blink': (left_eye_smooth, right_eye_smooth)}
            sio.emit('result_data', result_string, namespace='/model')
            put_nowait_drop_oldest(upstream_queue, (frame, landmarks, euler_angle))
            break # Explicitly process only the first detected face


def draw(color=(125, 255, 0), thickness=2):
    while True:
        frame, landmarks, euler_angle = upstream_queue.get()

        for p in np.round(landmarks).astype(int):
            cv2.circle(frame, tuple(p), 1, color, thickness, cv2.LINE_AA)

        # face_center = np.mean(landmarks, axis=0)
        # hp.draw_axis(frame, euler_angle, face_center)

        frame = cv2.resize(frame, (540, 450))

        cv2.imshow('result', frame)
        
        # fix press Q to stop capture video from camera
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        # cv2.waitKey(1)


draw_thread = Thread(target=draw)
draw_thread.start()

iris_thread = Thread(target=iris_localization)
iris_thread.start()

alignment_thread = Thread(target=face_alignment)
alignment_thread.start()

face_detection()
cap.release()
cv2.destroyAllWindows()
