import cv2
import numpy as np
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

def find_cube(frame, mask, elements):
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
                    elements_color += 1
                    cv2.circle(frame, (int(x), int(y)), int(rayon), [0, 0, 255], 2)

                    Xs.append(x)
                    Ys.append(y)

    return elements_color, frame, Xs, Ys

cap = cv2.VideoCapture(6)



for i in range(30):
    ret, frame = cap.read()

if not cap.isOpened():
    print("Cannot open camera")
    exit()

while True:

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
    elements_green, frame, Xs_green, Ys_Green = find_cube(frame, mask_green, elements_green)
    posX['green'] = Xs_green
    posY['green'] = Ys_Green

    elements_blue=0
    mask_blue = blue_hsv()
    detected_img = cv2.bitwise_and(frame, frame, mask= mask_blue)
    elements_blue, frame, Xs_blue, Ys_blue= find_cube(frame, mask_blue, elements_blue)
    posX['blue'] = Xs_blue
    posY['blue'] = Ys_blue
  
    '''
    elements_red=0
    mask_red = red_hsv()
    detected_red = cv2.bitwise_and(frame, frame, mask= mask_red)
    elements_red, frame = find_cube(frame, mask_red, elements_red)
    '''  
    
    elements_yellow=0
    mask_yellow = yellow_hsv()
    detected_yellow = cv2.bitwise_and(frame, frame, mask= mask_yellow)
    elements_yellow, frame, Xs_yellow, Ys_yellow= find_cube(frame, mask_yellow, elements_yellow)
    posX['yellow'] = Xs_yellow
    posY['yellow'] = Ys_yellow


    cv2.imshow('Contours Connexes', frame)
    
    if cv2.waitKey(1) == ord('q'):
        break

print(len(posX.values()))
print(len(posY.values()))

cv2.destroyAllWindows()

