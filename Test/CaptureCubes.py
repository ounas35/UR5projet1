import cv2
import numpy as np
import pyrealsense2 as rs
import math
import time

cap = cv2.VideoCapture(6)


for i in range(30):
    ret, frame = cap.read()


if not cap.isOpened():
    print("Cannot open camera")
    exit()

while True:

    ret, frame = cap.read()
    
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    cv2.imshow('frame', frame)
    
        # Contrôle de flux
    if cv2.waitKey(1) == ord('q'):
        cv2.imwrite('frame.jpg', frame)
        break

cv2.destroyAllWindows()
cap.release()