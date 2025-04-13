import numpy as np 
'''
import rtde_receive
import rtde_control
import dashboard_client

import cv2

robot_r = rtde_receive.RTDEReceiveInterface("10.2.30.60")
robot = rtde_control.RTDEControlInterface("10.2.30.60")
dashboard =dashboard_client.DashboardClient("10.2.30.60")

#Robot on the cubes
print("Put the robot on the top of the cubes")
print("Tap p to register the position")

while True :



    # Contrôle de flux
    if cv2.waitKey(1) == ord('p'):
        poseCubeUp = robot_r.getActualTCPPose()
        articulationsCubeUp = robot_r.getActualQ()
        break

articulationsGridUp = [0.13442106544971466, -1.585430924092428, 1.5986056327819824, -1.6099160353290003, -1.4244063536273401, -1.6908624807940882]
robot.moveJ(articulationsGridUp)
poseGridUp = robot_r.getActualTCPPose()

cap = cv2.VideoCapture(6)

for i in range(30):
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
    break

ret, background = cap.read()

cv2.imwrite('./Images/background.png', background)

im = cv2.imread('./Images/background.png')
cv2.imshow('Image de fond', im)
cv2.waitKey(0)

cv2.destroyAllWindows()

robot.moveJ(articulationsCubeUp)

'''
