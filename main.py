# import dependencies
import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import time
import mediapipe as mp

# Key points using MP Holistic

mp_holistic = mp.solutions.holistic  # Holisitc Model
mp_drawing = mp.solutions.drawing_utils  # Drawing utilities


def mediapip_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # COLOR CONVERSION BGR 2 RGB
    image.flags.writeable = False  # image is no longer Writeable
    results = model.process(image)  # make prediciton
    image.flags.writeable = True  # image is now writeable
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # COLOR CONVERSION RGB 2 BGR
    return image, results


def draw_styled_landmarks(image, results):
    # Draw Face Connections
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION,
                              mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1),
                              mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1))
    # Draw Pose Connections
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=1, circle_radius=1),
                              mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=1, circle_radius=1))
    # Draw Left Hand Connections
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks,
                              mp_holistic.HAND_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=1, circle_radius=1),
                              mp_drawing.DrawingSpec(color=(121, 44, 350), thickness=1, circle_radius=1))
    # Draw Right Hand Connections
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks,
                              mp_holistic.HAND_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=1, circle_radius=1),
                              mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=1, circle_radius=1))


cap = cv2.VideoCapture(1)
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        # read feed
        ret, frame = cap.read()

        # Make detections
        image, results = mediapip_detection(frame, holistic)

        # Draw Landmarks
        draw_styled_landmarks(image, results)

        # show to screen
        cv2.imshow('OpenCV Freed', image)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

print("Done")
cap.release()
cv2.destroyAllWindows()
