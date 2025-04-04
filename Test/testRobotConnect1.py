import numpy as np 
import rtde_receive
import rtde_control
import dashboard_client
import cv2 as cv
import math
import time

robot_r = rtde_receive.RTDEReceiveInterface("10.2.30.60")
robot = rtde_control.RTDEControlInterface("10.2.30.60")
dashboard =dashboard_client.DashboardClient("10.2.30.60")

pose = robot_r.getActualTCPPose()
articulations = robot_r.getActualQ()
articulations_degrees = [articulation/math.pi*180 for articulation in articulations]

print(pose)
print(articulations)
print([articulation/math.pi*180 for articulation in articulations])

time.sleep(10)
robot.moveJ(articulations)

