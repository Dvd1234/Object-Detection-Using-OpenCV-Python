# -*- coding: utf-8 -*-
"""
Created on Mon Dec 31 19:22:19 2018

@author: Deepak
"""
#this aims at checking if some new object has come in front of the camera and capture his image(If its an image) and time period for which it was in front of it.

import cv2

face_cascade = cv2.CascadeClassifier('C:/Users/wc/OpenCV_Face_Detection/cascades/data/haarcascade_frontalface_alt2.xml')

#here 0 is for the camera with the laptop while 1,2 3 etc would give the external webcams
cap=cv2.VideoCapture(0)

first_frame=None

object_detected=False

while True:
    object_detected=False
    check,frame=cap.read()
    gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gaussian_blur=cv2.GaussianBlur(gray,(21,21),0)
    
    #Store the first image
    if first_frame is None:
        first_frame=gaussian_blur
        continue;
        
    #Defining various threshold of detection and ignoring meagre ones
    delta_frame=cv2.absdiff(first_frame,gaussian_blur)
    thresh_delta=cv2.threshold(delta_frame,30,255,cv2.THRESH_BINARY)[1]
    thresh_delta=cv2.dilate(thresh_delta,None,iterations=0)
    
    #TO find contour
    (_,cnts,_)=cv2.findContours(thresh_delta.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    
    
    #To Reduce noise
    for contour in cnts:
        #checking the area of contours. Greater area gave better results for one big object detection
        if(cv2.contourArea(contour)<10000):
            continue;
        object_detected=True
        x,y,w,h =cv2.boundingRect(contour)
        font = cv2.FONT_HERSHEY_SIMPLEX
        color = (255, 255, 255)
        cv2.putText(frame, "Object Detected", (x,y), font, 0.5, color, 1, cv2.LINE_AA)
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
    
    if object_detected:
        
        #If there is a face in the object detect and save that
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
        for (x, y, w, h) in faces:
            roi_gray = gray[y:y+h, x:x+w] #(ycord_start, ycord_end)
            roi_color = frame[y:y+h, x:x+w]#region of interest
            color = (255, 0, 0) #BGR 0-255 
            stroke = 2
            end_cord_x = x + w
            end_cord_y = y + h
            cv2.rectangle(frame, (x, y), (end_cord_x, end_cord_y), color, stroke)
            img_item = "C:\\Users\\wc\\Pictures\\Saved Pictures\\object.png"
            cv2.imwrite(img_item, roi_color)
            	
    cv2.imshow('Camera Frame Python',frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

