import numpy as np
import cv2
import os

faceDetect = cv2.CascadeClassifier('./utils/haarcascade_frontalface_default.xml')
vid = cv2.VideoCapture(0,cv2.CAP_DSHOW)
# EigenFace and FisherFace require all images for training to be of equal dimensions
height = 215
width = 215

name=input("Enter name:")
name = name.lower()
dirname = './utils/dataset/'+name+'/'
print("Creating dataset for "+name.upper())
for i in range(21):
    ret,img = vid.read()
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = faceDetect.detectMultiScale(gray,1.3,5)
    for(x,y,w,h) in faces:
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        cv2.imwrite(dirname+str(i)+".jpg",gray[y:y+h,x:x+w])
        # cv2.imwrite(dirname+str(i)+".jpg",cv2.resize(gray[y:y+h,x:x+w], (width, height)))
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,225),2)
    cv2.imshow("Face",img)
    cv2.waitKey(1)
vid.release()
cv2.destroyAllWindows()