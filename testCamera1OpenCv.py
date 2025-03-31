import numpy as np
import cv2 as cv

image_path = "/home/robot/Documents/Projet/UR5projet1/images/image.png"

#-------------------FUNCTIONS-------------------

# Lower and higher HSV value for red color
def yellow_hsv():
    # H -> 0-13
    # S -> 200-255
    # V -> 50-160

    # define range of red color in HSV
    lower_hsv = np.array([20, 100, 30]) 
    higher_hsv = np.array([30, 255, 240])
    
    # generating mask for red color
    mask = cv.inRange(hsv_img, lower_hsv, higher_hsv)
    return mask

def red_hsv():
    # H -> 0-13
    # S -> 200-255
    # V -> 50-160

    # define range of red color in HSV
    lower_hsv = np.array([0, 0, 0]) 
    higher_hsv = np.array([19, 255, 255])
    
    # generating mask for red color
    mask = cv.inRange(hsv_img, lower_hsv, higher_hsv)
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
    mask = cv.inRange(hsv_img, lower_hsv, higher_hsv)
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
    mask = cv.inRange(hsv_img, lower_hsv, higher_hsv)
    return mask


#-------------------OPEN CAMERA-------------------
cap1 = cv.VideoCapture(4)
cap2 = cv.VideoCapture(6)

#-------------------TEST WITH ONE IMAGE-------------------
'''
for i in range (10):
    _, img = cap2.read()

cv.imwrite(image_path,img)

# read image
bgr_img = cv.imread(image_path)
# Convert HSV
hsv_img = cv.cvtColor(bgr_img, cv.COLOR_BGR2HSV)

mask = green_hsv()

detected_img1 = cv.bitwise_and(bgr_img, bgr_img, mask= mask)
cv.imwrite("/home/robot/Documents/Projet/UR5projet1/images/image_test_couleur.png", detected_img1)
'''

#-------------------TEST WITH THE VIDEO OF THE CAMERA-------------------

if not cap1.isOpened():
    print("Cannot open camera")
    exit()
while True:
    # Capture frame-by-frame
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()

    if not ret1:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    # Display the resulting frame
    #cv.imshow('frame1', frame1)
    #cv.imshow('frame2', frame2)


    hsv_img = cv.cvtColor(frame2, cv.COLOR_BGR2HSV)

    mask = red_hsv()

    detected_img = cv.bitwise_and(frame2, frame2, mask= mask)
    cv.imshow("detected image", detected_img)

    if cv.waitKey(1) == ord('q'):
        break


#-------------------CLOSE CAMERA-------------------
cap1.release()
cap2.release()
cv.destroyAllWindows()
