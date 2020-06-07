import numpy as np
import cv2
import os
import json

faceDetect = cv2.CascadeClassifier('./utils/haarcascade_frontalface_default.xml')
face_recognizer = cv2.face.LBPHFaceRecognizer_create()
# face_recognizer = cv2.face.EigenFaceRecognizer_create()
# face_recognizer = cv2.face.FisherFaceRecognizer_create()
face_recognizer.read("./utils/model/model.yml")

# EigenFace and FisherFace require all images for training to be of equal dimensions
width = 215
height = 215

with open('./utils/model/config.json') as f:
    data = json.load(f)

vid = cv2.VideoCapture(0,cv2.CAP_DSHOW)
while(True):
    ret,img = vid.read()
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = faceDetect.detectMultiScale(gray,1.3,5)
    for(x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,225),2)
        label,dist = face_recognizer.predict(gray[y:y+h,x:x+w])
        # label,dist = face_recognizer.predict(cv2.resize(gray[y:y+h,x:x+w],(width,height)))
        cv2.putText(img,str(data[str(label)]).upper(),(x,y+h+25),cv2.FONT_HERSHEY_PLAIN,1,(0, 255, 0),2)
    cv2.imshow("Face",img)
    if(cv2.waitKey(1)==ord('q')):
        break
vid.release()
cv2.destroyAllWindows()