import cv2
import numpy as np
import pyrealsense2 as rs
import math
import time

'''
for i in range(30):

    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        print(i)
'''

def red_hsv():
    # H -> 0-13
    # S -> 200-255
    # V -> 50-160

    # define range of red color in HSV
    lower_hsv = np.array([0, 70, 0]) 
    higher_hsv = np.array([19, 255, 235])
    
    # generating mask for red color
    mask = cv2.inRange(hsv_img, lower_hsv, higher_hsv)
    return mask

# Lower and higher HSV value for blue color
def blue_hsv():
    # H -> 80-240
    # S -> 140-240
    # V -> 100-170
    
    # define range of blue color in HSV
    lower_hsv = np.array([102, 70, 0])
    higher_hsv = np.array([120, 255, 255])
    
    # generating mask for blue color
    mask = cv2.inRange(hsv_img, lower_hsv, higher_hsv)
    return mask

def find_cube_grid(frame, mask, elements,square_color, taille_rayon):
    elements_color = 0
    Xs = []
    Ys = []
    elements, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    c = sorted(elements, key=cv2.contourArea)
    for region in c:
            ((x, y), rayon) = cv2.minEnclosingCircle(region)
            rect = cv2.minAreaRect(region)
        
            if(rayon > 7 and rayon < 50):
                #elements_color += 1

                # Extraire la couleur hvs de la région
                roi = frame[int(y)-int(rayon):int(y)+int(rayon), int(x)-int(rayon):int(x)+int(rayon)]
                
                #hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                hsv_roi = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

                # Définir les plages de couleur pour le bleu, le vert, le jaune et l'orange
                lower = np.array([102, 125, 0])
                upper = np.array([120, 255, 180])

                # Vérifier si la couleur de la région correspond à l'une des plages définies
                if cv2.inRange(hsv_roi, lower, upper).any():
                    # Vérifier la taille du rayon 
                    if rayon > taille_rayon :
                        #if x > box[0][0] and x < box[1][0] and y > box[0][1] and y < box[3][1]:
                        elements_color += 1
                        #print(f"Rayon : {rayon}")
                        cv2.circle(frame, (int(x), int(y)), int(rayon), square_color, 2)                # Square
                        cv2.circle(frame,(int(x), int(y)),5,square_color,2)                             # Centre
                        Xs.append(int(x))
                        Ys.append(int(y))

    return elements_color, frame, Xs, Ys

def find_grid(frame, mask, square_color, taille_rayon):
    box = []
    elements, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    c = sorted(elements, key=cv2.contourArea)
    for region in c:
            ((x, y), rayon) = cv2.minEnclosingCircle(region)
            rect = cv2.minAreaRect(region)

            if(rayon > 150 and rayon < 300):

                # Extraire la couleur hvs de la région
                roi = frame[int(y)-int(rayon):int(y)+int(rayon), int(x)-int(rayon):int(x)+int(rayon)]
                
                #hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                hsv_roi = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

                # Définir les plages de couleur pour le bleu, le vert, le jaune et l'orange
                lower = np.array([102, 125, 0])
                upper = np.array([120, 255, 180])

                # Vérifier si la couleur de la région correspond à l'une des plages définies
                if cv2.inRange(hsv_roi, lower, upper).any():
                    # Vérifier la taille du rayon 
                    if rayon > taille_rayon : 
                        #print(f"Rayon : {rayon}")
                        
                        box = cv2.boxPoints(rect)
                        box = np.intp(box)
                        box[0] = (x - rayon*math.sqrt(2)/2 , y - rayon*math.sqrt(2)/2) 
                        box[1] = (x + rayon*math.sqrt(2)/2,  y - rayon*math.sqrt(2)/2 )
                        box[2] = (x + rayon*math.sqrt(2)/2,  y + rayon*math.sqrt(2)/2 )
                        box[3] = (x - rayon*math.sqrt(2)/2,  y + rayon*math.sqrt(2)/2 )
                        box = np.intp(box)
                        cv2.drawContours(frame,[box],0,square_color,2)

    return frame, box

"""
Utilisation de la camera intel realsense
"""
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

"""
Estimation de position X, Y, Z par la camera realsense dans le repere camera
"""
def positionXYZ(x, y, camera_configurations):
    # x et y en pixel
    [pipeline, align, depth_scale, color_intrinsics,color_extrinsics] = camera_configurations 

    time.sleep(1)
    frames = pipeline.wait_for_frames()
    aligned_frames = align.process(frames)

    # Get aligned frames
    aligned_depth_frame = aligned_frames.get_depth_frame()  # aligned_depth_frame is a 640x480 depth image
    color_frame = aligned_frames.get_color_frame()
    depth = aligned_depth_frame.get_distance(x, y)
    # X, Y, Z = rs.rs2_deproject_pixel_to_point(intr, pixel, depth)
    point = rs.rs2_deproject_pixel_to_point(color_intrinsics, [x, y], depth)
    # en mètres m
    point=[point[0]*1000, point[1]*1000, point[2]*1000]
    return point

cap = cv2.VideoCapture(6)

taille_rayon = 30

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

    posX = {}
    posY = {}

    hsv_img = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    elements_blue=0
    mask_blue = blue_hsv()
    detected_img = cv2.bitwise_and(frame, frame, mask= mask_blue)

    color_green = [0,255,0]
    color_blue = [255,0,0]
    '''
    frame, box= find_grid(frame, mask_blue, color_green,taille_rayon)

    
    if not [box]:
        print("Invalid box detected. Skipping cube detection.")
        print(f"Detected box: {box}")
    else:
    '''
    elements_blue, frame, Xs_blue, Ys_blue = find_cube_grid(frame, mask_blue, elements_blue, color_blue, taille_rayon)

    cv2.imshow('Contours Connexes', frame)

    # Contrôle de flux
    if cv2.waitKey(1) == ord('q'):
        break


for i in range(30):
    ret, frame = cap.read()
cv2.destroyAllWindows()
cap.release()
'''
if not [box]:
    print("Invalid box detected. Skipping cube detection.")
else:
'''
elements_blue, frame, Xs_blue, Ys_blue = find_cube_grid(frame, mask_blue, elements_blue, color_blue, taille_rayon)


posX['blue'] = Xs_blue
posY['blue'] = Ys_blue

# Messure de distance par rapport la caméra
Xs_values = []
Ys_values = []
for key in posX.keys():
    posx_values = posX[key]
    posy_values = posY[key]
    Xs_values += posx_values
    Ys_values += posy_values

print(f"X_values : {Xs_values}")
print(f"Y_values : {Ys_values}")

Pos = np.empty((len(Xs_values),3))
camera_calibration = initialize_device()
for i in range(len(Xs_values)):

    Pos[i][0],Pos[i][1],Pos[i][2] = positionXYZ(Xs_values[i],Ys_values[i],camera_calibration)

print(Pos)
print(len(Pos))