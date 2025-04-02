import cv2
import numpy as np
import pyrealsense2 as rs

'''
for i in range(30):

    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        print(i)
'''

# Lower and higher HSV value for red color
def yellow_hsv():
    # H -> 0-13
    # S -> 200-255
    # V -> 50-160

    # define range of red color in HSV
    lower_hsv = np.array([20, 100, 30]) 
    higher_hsv = np.array([30, 255, 240])
    
    # generating mask for red color
    mask = cv2.inRange(hsv_img, lower_hsv, higher_hsv)
    return mask

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
    lower_hsv = np.array([102, 125, 0])
    higher_hsv = np.array([120, 255, 180])
    
    # generating mask for blue color
    mask = cv2.inRange(hsv_img, lower_hsv, higher_hsv)
    return mask

# Lower and higher HSV value for blue color
def green_hsv():
    # H -> 70-85
    # S -> 105-230
    # V -> 40-160
    
    # define range of blue color in HSV
    lower_hsv = np.array([60, 105, 40])
    higher_hsv = np.array([85, 230, 200])
    
    # generating mask for blue color
    mask = cv2.inRange(hsv_img, lower_hsv, higher_hsv)
    return mask

def find_cube(frame, mask, elements,square_color, taille_rayon):
    elements_color = 0
    Xs = []
    Ys = []
    elements, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    c = sorted(elements, key=cv2.contourArea)
    for region in c:
            ((x, y), rayon) = cv2.minEnclosingCircle(region)
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
                        elements_color += 1
                        cv2.circle(frame, (int(x), int(y)), int(rayon), square_color, 2)                # Square
                        cv2.circle(frame,(int(x), int(y)),5,square_color,2)                             # Centre
                        Xs.append(int(x))
                        Ys.append(int(y))

    return elements_color, frame, Xs, Ys

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
def positionXYZ(x, y):
    # x et y en pixel
    pipeline, align, depth_scale, color_intrinsics,color_extrinsics = initialize_device()

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



for i in range(30):
    ret, frame = cap.read()

if not cap.isOpened():
    print("Cannot open camera")
    exit()

finished = False    

taille_rayon = 20

while True:

    if not finished:
        posX = {}
        posY = {}

        ret, frame = cap.read()
        
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        hsv_img = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        elements_green=0
        mask_green = green_hsv()
        detected_img = cv2.bitwise_and(frame, frame, mask= mask_green)
        color_green = [0,255,0]
        elements_green, frame, Xs_green, Ys_Green = find_cube(frame, mask_green, elements_green,color_green,taille_rayon)
        posX['green'] = Xs_green
        posY['green'] = Ys_Green
        
        elements_blue=0
        mask_blue = blue_hsv()
        detected_img = cv2.bitwise_and(frame, frame, mask= mask_blue)
        color_blue = [255,0,0]
        elements_blue, frame, Xs_blue, Ys_blue= find_cube(frame, mask_blue, elements_blue,color_blue,taille_rayon)
        posX['blue'] = Xs_blue
        posY['blue'] = Ys_blue
    
        '''
        elements_red=0
        mask_red = red_hsv()
        detected_red = cv2.bitwise_and(frame, frame, mask= mask_red)
        color_red = [0,0,255]
        elements_red, frame, Xs_red, Ys_red = find_cube(frame, mask_red, elements_red,color_red)
        posX['red'] = Xs_red
        posY['red'] = Ys_red  
        '''
        
        elements_yellow=0
        mask_yellow = yellow_hsv()
        detected_yellow = cv2.bitwise_and(frame, frame, mask= mask_yellow)
        color_yellow = [0,255,255]
        elements_yellow, frame, Xs_yellow, Ys_yellow= find_cube(frame, mask_yellow, elements_yellow,color_yellow,taille_rayon)
        posX['yellow'] = Xs_yellow
        posY['yellow'] = Ys_yellow
        

        cv2.imshow('Contours Connexes', frame)
    
    else :
        # Messure de distance par rapport la caméra
        #for Xs, Ys in zip(Xs_values, Ys_values) :
        #    print(Xs,Ys) 
        Xs_values = []
        Ys_values = []
        for key in posX.keys():
            posx_values = posX[key]
            posy_values = posY[key]
            Xs_values += posx_values
            Ys_values += posy_values

        
        Pos = np.empty((len(Xs_values),3))
        for i in range(len(Xs_values)):
            Pos[i][0],Pos[i][1],Pos[i][2]  = positionXYZ(Xs_values[i],Ys_values[i])
          
        break       
        
        # Contrôle de flux
    if cv2.waitKey(1) == ord('q'):
        finished = True
        cv2.destroyAllWindows()
        cap.release()


#print(len(posX.values()))
#print(len(posY.values()))
print(Pos)
cv2.destroyAllWindows()

