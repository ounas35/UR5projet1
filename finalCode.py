import cv2
import numpy as np
import pyrealsense2 as rs
import numpy as np 
import rtde_receive
import rtde_control
import dashboard_client
import cv2
import math
import time
import random
from scipy.spatial.transform import Rotation as R

#-------------------FUNCTIONS-------------------

# Lower and higher HSV value for yellow color
def yellow_hsv():

    # define range of yellow color in HSV
    lower_hsv = np.array([20, 100, 30]) 
    higher_hsv = np.array([30, 255, 240])
    
    # generating mask for yellow color
    mask = cv2.inRange(hsv_img, lower_hsv, higher_hsv)
    return mask


# Lower and higher HSV value for red color
def red_hsv():

    # define range of red color in HSV
    lower_hsv = np.array([0, 70, 0]) 
    higher_hsv = np.array([19, 255, 235])
    
    # generating mask for red color
    mask = cv2.inRange(hsv_img, lower_hsv, higher_hsv)
    return mask


# Lower and higher HSV value for blue color
def blue_hsv():
    
    # define range of blue color in HSV
    lower_hsv = np.array([102, 125, 0])
    higher_hsv = np.array([120, 255, 180])
    
    # generating mask for blue color
    mask = cv2.inRange(hsv_img, lower_hsv, higher_hsv)
    return mask


# Lower and higher HSV value for green color
def green_hsv():
    
    # define range of green color in HSV
    lower_hsv = np.array([60, 105, 40])
    higher_hsv = np.array([85, 230, 200])
    
    # generating mask for green color
    mask = cv2.inRange(hsv_img, lower_hsv, higher_hsv)
    return mask


# Finds and marks colored circular objects based on size and HSV color range
def find_cube(frame, mask, elements,square_color, taille_rayon):
    elements_color = 0
    Xs = []
    Ys = []
    # Find contours from the mask
    elements, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # Sort contours by area (small to large)
    c = sorted(elements, key=cv2.contourArea)
    for region in c:
            ((x, y), rayon) = cv2.minEnclosingCircle(region)
            # Filter by radius size
            if(rayon > 7 and rayon < 50):

                # Extract a region of interest (ROI) around the detected circle
                roi = frame[int(y)-int(rayon):int(y)+int(rayon), int(x)-int(rayon):int(x)+int(rayon)]
                
                # Convert ROI (or full frame) to HSV
                hsv_roi = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

                # Define HSV range 
                lower = np.array([102, 125, 0])
                upper = np.array([120, 255, 180])

                # Check if the region contains pixels in the color range
                if cv2.inRange(hsv_roi, lower, upper).any():
                    # Draw the detected circle and center
                    if rayon > taille_rayon : 
                        elements_color += 1
                        cv2.circle(frame, (int(x), int(y)), int(rayon), square_color, 2)               
                        cv2.circle(frame,(int(x), int(y)),5,square_color,2)                            
                        Xs.append(int(x))
                        Ys.append(int(y))

    return elements_color, frame, Xs, Ys


# Using the Intel RealSense camera
def initialize_device():
    # Create a pipeline
    pipeline = rs.pipeline()
    config = rs.config()

    pipeline_wrapper = rs.pipeline_wrapper(pipeline)
    pipeline_profile = config.resolve(pipeline_wrapper)
    device = pipeline_profile.get_device()


    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    # Start streaming
    profile = pipeline.start(config)

    # Get stream profile and camera intrinsics
    color_profile = rs.video_stream_profile(profile.get_stream(rs.stream.color))
    color_intrinsics = color_profile.get_intrinsics()
    color_extrinsics = color_profile.get_extrinsics_to(color_profile)
   
    # Getting the depth sensor's depth scale (see rs-align example for explanation)
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()

    # Create an align object
    # rs.align allows us to perform alignment of depth frames to others frames
    # The "align_to" is the stream type to which we plan to align depth frames.
    align_to = rs.stream.color
    align = rs.align(align_to)

    return pipeline, align, depth_scale,color_intrinsics,color_extrinsics


#X, Y, Z position estimation by the realsense camera in the camera frame
def positionXYZ(x, y):

    # Initialize the RealSense camera and its settings
    pipeline, align, depth_scale, color_intrinsics,color_extrinsics = initialize_device()

    time.sleep(1)
    frames = pipeline.wait_for_frames()
    aligned_frames = align.process(frames)

    # Get aligned frames
    aligned_depth_frame = aligned_frames.get_depth_frame()  
    color_frame = aligned_frames.get_color_frame()
    depth = aligned_depth_frame.get_distance(x, y)
    point = rs.rs2_deproject_pixel_to_point(color_intrinsics, [x, y], depth)

    # Convert the point to millimeters (RealSense gives meters by default)
    point=[point[0]*1000, point[1]*1000, point[2]*1000]
    pipeline.stop()
    return point

# Find the grid of cubes in the image and mark them
def find_cube_grid(frame, mask, elements, square_color, taille_rayon, box):
    elements_color = 0 
    Xs = []  
    Ys = []  
    
    # Find contours in the mask (regions of interest)
    elements, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    c = sorted(elements, key=cv2.contourArea)  # Sort contours by area (from largest to smallest)
    
    for region in c:
            # Get the enclosing circle for each contour (region)
            ((x, y), rayon) = cv2.minEnclosingCircle(region)
            rect = cv2.minAreaRect(region) 
        
            # Only consider regions with a reasonable size
            if (rayon > 7 and rayon < 50):

                # Extract the region of interest (ROI) from the frame based on the bounding circle
                roi = frame[int(y) - int(rayon):int(y) + int(rayon), int(x) - int(rayon):int(x) + int(rayon)]
                
                # Convert the frame to HSV for color detection
                hsv_roi = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

                # Define color ranges for detecting certain colors (blue in this case)
                lower = np.array([102, 125, 0])
                upper = np.array([120, 255, 180])

                # Check if the color of the region matches the defined range
                if cv2.inRange(hsv_roi, lower, upper).any():
                    # Check if the radius is large enough to be considered
                    if rayon > taille_rayon:
                        # Check if the detected point is inside the specified box
                        if x > box[0][0] and x < box[1][0] and y > box[0][1] and y < box[3][1]:
                            elements_color += 1  # Increment color-matching element counter
                            # Draw a circle around the detected element and mark its center
                            cv2.circle(frame, (int(x), int(y)), int(rayon), square_color, 2) 
                            cv2.circle(frame, (int(x), int(y)), 5, square_color, 2)  
                            Xs.append(int(x))  
                            Ys.append(int(y))  

    return elements_color, frame, Xs, Ys  # Return the number of elements found and their positions

# Find the grid of cubes in the image and mark them
def find_grid(frame, mask, square_color, taille_rayon):
    box = [] 
    # Find contours in the mask
    elements, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    c = sorted(elements, key=cv2.contourArea)  # Sort contours by area
    
    for region in c:
            ((x, y), rayon) = cv2.minEnclosingCircle(region)  
            rect = cv2.minAreaRect(region)  

            # Only consider contours within a certain radius range (likely for larger elements)
            if (rayon > 150 and rayon < 300):

                # Extract the region of interest (ROI) from the frame based on the bounding circle
                roi = frame[int(y) - int(rayon):int(y) + int(rayon), int(x) - int(rayon):int(x) + int(rayon)]
                
                # Convert the frame to HSV for color detection
                hsv_roi = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

                # Define color ranges for detecting certain colors (blue in this case)
                lower = np.array([102, 125, 0])
                upper = np.array([120, 255, 180])

                # Check if the color of the region matches the defined range
                if cv2.inRange(hsv_roi, lower, upper).any():
                    # Check if the radius is large enough to be considered
                    if rayon > taille_rayon: 
                        #print(f"Radius: {rayon}")
                        
                        # Get the box points for the bounding rectangle around the contour
                        box = cv2.boxPoints(rect)
                        box = np.intp(box)
                        # Adjust the box points based on the radius
                        box[0] = (x - rayon * math.sqrt(2) / 2 , y - rayon * math.sqrt(2) / 2) 
                        box[1] = (x + rayon * math.sqrt(2) / 2,  y - rayon * math.sqrt(2) / 2 )
                        box[2] = (x + rayon * math.sqrt(2) / 2,  y + rayon * math.sqrt(2) / 2 )
                        box[3] = (x - rayon * math.sqrt(2) / 2,  y + rayon * math.sqrt(2) / 2 )
                        box = np.intp(box)
                        # Draw the contours of the box
                        cv2.drawContours(frame, [box], 0, square_color, 2)

    return frame, box 

#Compute the inverse kinematics for a given end-effector pose (x, y, z) and orientation (alpha, beta, gamma)
def compute_inverse_kinematics(x, y, z, alpha, beta, gamma):
    # Step 1: Compute the homogeneous transformation (position + orientation)
    r = R.from_euler('xyz', [alpha, beta, gamma])
    R06 = r.as_matrix()
    
    # Position of the tool (TCP)
    p06 = np.array([x, y, z])

    # Compute the wrist center position (remove the last joint offset)
    pw = p06 - d6 * R06[:, 2]  # d6 along z-axis of the tool

    # q1: base rotation angle
    q1 = np.arctan2(pw[1], pw[0])

    # Planar distances for solving q2 and q3
    r1 = np.sqrt(pw[0]**2 + pw[1]**2)
    r2 = pw[2] - d1
    r3 = np.sqrt(r1**2 + r2**2)

    # Law of cosines to solve for q3
    cos_q3 = (r3**2 - a2**2 - a3**2) / (2 * a2 * a3)
    q3 = np.arccos(np.clip(cos_q3, -1.0, 1.0))  # Clamp to avoid domain error

    # q2 based on triangle geometry
    q2 = np.arctan2(r2, r1) - np.arctan2(a3 * np.sin(q3), a2 + a3 * np.cos(q3))

    # Helper functions for rotation matrices
    def Rz(theta): return np.array([[np.cos(theta), -np.sin(theta), 0],
                                     [np.sin(theta),  np.cos(theta), 0],
                                     [0, 0, 1]])
    def Ry(theta): return np.array([[ np.cos(theta), 0, np.sin(theta)],
                                     [0, 1, 0],
                                     [-np.sin(theta), 0, np.cos(theta)]])

    # Compute R03 using forward kinematics up to joint 3
    R03 = Rz(q1) @ Ry(q2) @ Ry(q3)

    # Compute R36 = R03.T * R06 (remaining rotation from joint 4 to 6)
    R36 = R03.T @ R06

    # Solve for q4, q5, q6 from R36
    q5 = np.arccos(np.clip(R36[2, 2], -1.0, 1.0))
    q4 = np.arctan2(R36[1, 2], R36[0, 2])
    q6 = np.arctan2(R36[2, 1], -R36[2, 0])

    return np.array([q1, q2, q3, q4, q5, q6])


# Find a random cube position and compute the inverse kinematics for it
def find_and_compute_inverse_kinematics(robot_position, cubes_positions, camera_distance, offset):

    # Find a random cube position from the list
    random_index = random.randint(0, len(cubes_positions) - 1)
    cube_position = cubes_positions.pop(random_index)  # Delete the position in the list

    # Create a pose for the cube in the robot's coordinate system
    pose_cube_robot = np.array(robot_position) + np.array(camera_distance) + np.array(cube_position) + np.array([0,0,offset])

    # Compute the inverse kinematics to get the joint angles of the robot
    joint_angles = compute_inverse_kinematics(pose_cube_robot)

    # Return the joint angles, the pose of the cube and the cube position
    return joint_angles, pose_cube_robot, cube_position


#-------------------INITIALISATION-------------------

# Parameters
taille_rayon = 20  
articulationsCubeUp =[-1.4837225119220179, -2.046366516743795, 1.0656824111938477, -0.6960395018206995, -1.5023983160602015, -3.102428976689474]
articulationsGridUpCalibration = [-0.3859136740313929, -2.123759094868795, 1.5018048286437988, -0.9511588255511683, -1.5231474081622522, -1.937965218220846]
articulationsGridUpCube = [0.22260510921478271, -1.8566058317767542, 1.3934135437011719, -1.1440780798541468, -1.5327971617328089, -1.3406441847430628]
# UR5 DH parameters (in meters)
d1 = 0.089159
a2 = -0.425
a3 = -0.39225
d4 = 0.10915
d5 = 0.09465
d6 = 0.0823

# Configuration
robot_r = rtde_receive.RTDEReceiveInterface("10.2.30.60") 
robot = rtde_control.RTDEControlInterface("10.2.30.60")  
dashboard = dashboard_client.DashboardClient("10.2.30.60")  

cap = cv2.VideoCapture(6)  # Open camera for image capture
robot.moveJ(articulationsCubeUp)  # Move robot to predefined position over cubes
poseCubeUp = robot_r.getActualTCPPose()  # Get current pose of the robot

# Initial image capture from the camera
for i in range(30):
    ret, frame = cap.read()

if not cap.isOpened():
    print("Cannot open camera")
    exit()

# Robot on the cubes
print("Put the robot on the top of the cubes")  
print("Tap q to register the position of the camera on the grid")  

# Main loop for cube detection
while True:
    ret, frame = cap.read()
    
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")  
        break

    posX = {}
    posY = {}

    hsv_img = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)  

    # Cube detection for green color
    elements_green = 0
    mask_green = green_hsv() 
    detected_img = cv2.bitwise_and(frame, frame, mask=mask_green)  
    color_green = [0, 255, 0]  
    elements_green, frame, Xs_green, Ys_Green = find_cube(frame, mask_green, elements_green, color_green, taille_rayon)

    # Cube detection for blue color
    elements_blue = 0
    mask_blue = blue_hsv()  
    detected_img = cv2.bitwise_and(frame, frame, mask=mask_blue) 
    color_blue = [255, 0, 0]  
    elements_blue, frame, Xs_blue, Ys_blue = find_cube(frame, mask_blue, elements_blue, color_blue, taille_rayon)

    # Cube detection for yellow color
    elements_yellow = 0
    mask_yellow = yellow_hsv() 
    detected_yellow = cv2.bitwise_and(frame, frame, mask=mask_yellow) 
    color_yellow = [0, 255, 255] 
    elements_yellow, frame, Xs_yellow, Ys_yellow = find_cube(frame, mask_yellow, elements_yellow, color_yellow, taille_rayon)

    # Display the result with contours
    cv2.imshow('Contours Connexes', frame)

    # Flow control: register the camera's position when 'q' is pressed
    if cv2.waitKey(1) == ord('q'):
        poseCubeUp = robot_r.getActualTCPPose()  # Get the robot's current tool center point pose
        articulationsCubeUp = robot_r.getActualQ()  # Get the robot's current joint positions
        break

# Store the detected positions for blue and yellow cubes
posY['blue'] = Ys_blue
elements_yellow, frame, Xs_yellow, Ys_yellow = find_cube(frame, mask_yellow, elements_yellow, color_yellow, taille_rayon)
posX['yellow'] = Xs_yellow
posY['yellow'] = Ys_yellow

# Release the camera for reuse in later steps
cap.release()

# Measure distance from the camera
Xs_values = []
Ys_values = []
for key in posX.keys():
    posx_values = posX[key]
    posy_values = posY[key]
    Xs_values += posx_values
    Ys_values += posy_values

PosCube = np.empty((len(Xs_values), 3))  
for i in range(len(Xs_values)):
    PosCube[i][0], PosCube[i][1], PosCube[i][2] = positionXYZ(Xs_values[i], Ys_values[i]) 

print(len(PosCube)) 

# Move the robot to grid position for cube placement
robot.moveJ(articulationsGridUpCube)
poseGridUpCube = robot_r.getActualTCPPose()  

# Open the camera again for the next detection cycle
cap = cv2.VideoCapture(6)

taille_rayon = 30  

# Capture frames from the camera
for i in range(30):
    ret, frame = cap.read()

if not cap.isOpened():
    print("Cannot open camera")
    exit()

# Main loop for grid detection and cube positioning
while True:
    ret, frame = cap.read()
    
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    posX = {}
    posY = {}

    hsv_img = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Cube detection for blue color and grid box finding
    elements_blue = 0
    mask_blue = blue_hsv()
    detected_img = cv2.bitwise_and(frame, frame, mask=mask_blue)

    color_green = [0, 255, 0]  
    frame, box = find_grid(frame, mask_blue, color_green, taille_rayon) 

    color_blue = [255, 0, 0]  # Blue color for cube detection
    if not [box]:
        print("Invalid box detected. Skipping cube detection.") 
        print(f"Detected box: {box}")  
    else:
        elements_blue, frame, Xs_blue, Ys_blue = find_cube_grid(frame, mask_blue, elements_blue, color_blue, taille_rayon, box)

    # Display the result with grid and cube detections
    cv2.imshow('Contours Connexes', frame)

    # Flow control: exit loop when 'q' is pressed
    if cv2.waitKey(1) == ord('q'):
        break

# Capture the last frames from the camera
for i in range(30):
    ret, frame = cap.read()
cv2.destroyAllWindows() 
cap.release()  

if not [box]:
    print("Invalid box detected. Skipping cube detection.")  # Handle invalid box
else:
    elements_blue, frame, Xs_blue, Ys_blue = find_cube_grid(frame, mask_blue, elements_blue, color_blue, taille_rayon, box)

# Store the positions for the detected blue cubes
posX['blue'] = Xs_blue
posY['blue'] = Ys_blue

# Measure distance from the camera for blue cubes
Xs_values = []
Ys_values = []
for key in posX.keys():
    posx_values = posX[key]
    posy_values = posY[key]
    Xs_values += posx_values
    Ys_values += posy_values

print(f"X_values : {Xs_values}")  # Print X positions
print(f"Y_values : {Ys_values}")  # Print Y positions

# Get the 3D positions of the blue cubes
PoseGrid = np.empty((len(Xs_values), 3))
for i in range(len(Xs_values)):
    PoseGrid[i][0], PoseGrid[i][1], PoseGrid[i][2] = positionXYZ(Xs_values[i], Ys_values[i])

print(PoseGrid)  
print(len(PoseGrid))  

#-------------------LOOP-------------------

camera_distance = [0.1, 0.1, 0.1] # Distance between the camera and the gripper
height_cube = 0.05 # Height of the cube

#--------Close the gripper--------

# Move the cubes to the 9 grid positions
for i in range(9):

    # Move on the top of the cubes 
    robot.moveJ(articulationsCubeUp) 
    
    #Find the cube position of one random cube and compute the inverse kinematics
    joint_angles, pose_cube_robot, cube_position = find_and_compute_inverse_kinematics(poseCubeUp, PosCube, camera_distance, 0.0)

    # Move to the cube position chosen randomly
    robot.moveJ(joint_angles)
    #take the cube
    #--------Open the gripper--------

    # Move on the top of the cubes 
    robot.moveJ(articulationsCubeUp) 

    # Move on the top of the grid
    robot.moveJ(articulationsGridUpCube)

    #Find the grid position of one random position and compute the inverse kinematics
    joint_angles, pose_cube_robot, cube_position = find_and_compute_inverse_kinematics(poseGridUpCube, PoseGrid, camera_distance, height_cube)

    # Move to the position chosen randomly on the grid
    robot.moveJ(joint_angles)
    #release the cube
    #--------Close the gripper--------

    # Move on the top of the grid
    robot.moveJ(articulationsGridUpCube)











