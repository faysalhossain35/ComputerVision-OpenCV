# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 12:08:28 2019

@author: Faysal
"""

import cv2

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier('haarcascade_smile.xml')


def detection(gray,frame):
    face = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    for x, y, w, h in face:
        cv2.rectangle(frame, (x, y), (x+w, y+h),(0,255,255),5)
        roi_gray = gray[x:x+w,y:y+h]
        roi_color = frame[x:x+w,y:y+h]
        
        eye = eye_cascade.detectMultiScale(roi_gray, 1.1, 2)
        
        for ex, ey, ew, eh in eye:
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh),(0,0,255),2)

        smile = smile_cascade.detectMultiScale(roi_gray, 1.1, 2)
        
        for sx, sy, sw, sh in smile:
            cv2.rectangle(roi_color, (sx, sy), (sx+sw, sy+sh),(0,255,0),2)
    
    return frame

# Capturing vedio by camera

video_cap = cv2.VideoCapture(0)
while True:
    _,frame = video_cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detect_face = detection(gray,frame)
    cv2.imshow('Video', detect_face)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
video_cap.release()
cv2.destroyAllWindows()


        
       

        