import cv2
import numpy as np

 
facedetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml') #

video = cv2.VideoCapture(0)

recognizer = cv2.face.LBPHFaceRecognizer_create()

recognizer.read("recognizer\\Training.ynl")

Id = 0

font = cv2.FONT_HERSHEY_SIMPLEX

while True:
    check , frame = video.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(gray,1.3,5)
    for x,y,h,w in faces:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (50,50,255),2)
        cv2.rectangle(frame, (x,y-40),(x+w, y), (50,50,255),-2)
        Id,conf = recognizer.predict(gray[y:y+h, x:x+w])
        if Id==1:
            Id = "ABED"
        cv2.putText(frame,str(Id),(x,y-10),font,1,(255,255,255),3) #can change the font    
    cv2.imshow("Deteced Face",frame)
    k = cv2.waitKey(1)
    if k==ord('q'):
        break

video.release()    


        


        








