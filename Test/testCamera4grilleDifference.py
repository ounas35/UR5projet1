import numpy as np 
import rtde_receive
import rtde_control
import dashboard_client
import cv2

robot_r = rtde_receive.RTDEReceiveInterface("10.2.30.60")
robot = rtde_control.RTDEControlInterface("10.2.30.60")
dashboard =dashboard_client.DashboardClient("10.2.30.60")

cap = cv2.VideoCapture(6)

# 1. Le chargement de l'image de fond
for i in range(30):
    ret, frame = cap.read()

im = cv2.imread('./Images/background.png')
cv2.imshow('Image de fond', im)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    # 3. Differences entre l'image de fond et la vidéo
    diff = cv2.absdiff(im, frame)
    # 4. Affichage (dans deux fenêtres différentes) de l'image de la vidéo et de l'image différence.
    cv2.imshow('Image de la video', frame)
    cv2.imshow('Image diffrence', diff)
    # 5. Conversion de l'image différence en niveaux de gris et seuillage
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    '''
    # 6. Procéder (par binarisation) à la création du masque de chaque objet :
    _, thresh = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)
    # 7. Nettoyer (supprimer les petites régions, le bruit etc.) ce masque avec des opérateurs morpholgiques (filtre médian, dilatation, fermeture etc.)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)) 
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    # 8. segmenter ce masque en plusieurs régions de pixels connexes
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # 9. Afficher tous les contours de ces régions en vert sur l'image
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        # Identification de piètons et voitures pour sa taille
        if cv2.contourArea(contour) > 500:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        elif cv2.contourArea(contour) > 15:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.imshow('Image de la vidéo', frame)
        cv2.waitKey(1)
    '''
    cv2.waitKey(0)
cap.release()