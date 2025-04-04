import numpy as np 
import rtde_receive
import rtde_control
import dashboard_client
import cv2 as cv
import math
robot_r = rtde_receive.RTDEReceiveInterface("10.2.30.60")
robot = rtde_control.RTDEControlInterface("10.2.30.60")
dashboard =dashboard_client.DashboardClient("10.2.30.60")

pose = robot_r.getActualTCPPose()
articulations = robot_r.getActualQ()
articulations_degrees = [articulation/math.pi*180 for articulation in articulations]

print(pose)
print(articulations)
print([articulation/math.pi*180 for articulation in articulations])

#poseBase = [0.13442106544971466, -1.585430924092428, 1.5986056327819824, -1.6099160353290003, -1.4244063536273401, -1.6908624807940882]

#robot.moveJ(poseBase)

