import cv2
import numpy as np


face_cascade=cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")

cap = cv2.VideoCapture("example2.mp4")

while True:
    _,frame = cap.read()
 
    detections = face_cascade.detectMultiScale(frame,scaleFactor=1.1,minNeighbors=6)

    for face in detections:
        x,y,w,h = face
        frame[y:y+h,x:x+w] = cv2.GaussianBlur(frame[y:y+h,x:x+w],(15,15),cv2.BORDER_DEFAULT)
        
        cv2.imshow("output",frame)
        if cv2.waitKey(1) == 27:
            break
cap.release()
cv2.destroyAllWindows()
