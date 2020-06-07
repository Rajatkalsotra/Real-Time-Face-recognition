import os
import cv2
import numpy as np
import json

faceDetect = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')
data = {}
dirname = "./utils/dataset/"

def getTrainingData():
	dirs = os.listdir(dirname)
	faces,labels = [],[]
	for i in range(len(dirs)):
		imgs = os.listdir(dirname+dirs[i])
		for img in imgs:
			face = cv2.imread(dirname+dirs[i]+'/'+img,0)
			faces.append(face)
			labels.append(i)
			data[i] = dirs[i]
	return faces, labels

faces,labels = getTrainingData()

if not os.path.exists('./utils/model'):
            os.makedirs('./utils/model')
with open('./utils/model/config.json', 'w') as f:
    json.dump(data, f)

print("Training Face recognition model")

face_recognizer = cv2.face.LBPHFaceRecognizer_create()
# face_recognizer = cv2.face.EigenFaceRecognizer_create()
# face_recognizer = cv2.face.FisherFaceRecognizer_create()

face_recognizer.train(faces, np.array(labels))
face_recognizer.save('./utils/model/model.yml')

print("Model trained")
print("Total faces:"+str(len(data)))