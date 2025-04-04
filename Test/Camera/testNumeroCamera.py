import cv2
import numpy as np
import pyrealsense2 as rs
import math
import time

for i in range(30):
    cap = cv2.VideoCapture(i)

    if cap.isOpened():
        print(i)
    else:
        print("Camera not found")