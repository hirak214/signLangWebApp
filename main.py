# import dependencies
import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import time
import mediapipe as mp

# Keypoints using MP Holistic
cap = cv2.VideoCapture(1)
while cap.isOpened():
    ret, frame = cap.read()
    cv2.imshow('OpenCV Freed', frame)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()