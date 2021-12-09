# -*- coding: utf-8 -*-
"""
Created on Fri Nov 26 21:34:13 2021

@author: dell
"""

import cv2
import numpy as np

haar_cascade = cv2.CascadeClassifier('C:/Users/dell/c,c++/.vscode/11.xml/haar_faces.xml')
people =['alexander dadario', 'cute girl', 'shirleu sethia', 'shruti', 'tamana']
features = np.load('features.npy',allow_pickle=True)
labels = np .load('labels.npy', allow_pickle= True)


face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.read(r'C:\Users\dell\PycharmProjects\pythonProject3\face_trained.yml')

image = cv2.imread(r'C:/Users/dell/Desktop/7.jpg')
cv2.imshow("image",image)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow("gray",gray)

face_rect  = haar_cascade.detectMultiScale(gray,scaleFactor=1.1, minNeighbors=5)

for(x,y,w,h) in face_rect:
    face_roi = gray[x:x+w , y:y+h]

    labels ,confidence     =      face_recognizer.predict(face_roi)
    print(f'labels = {people[labels]} with a confidence of {confidence} ')

    cv2.putText(image, str(people[labels]),(30,30) ,cv2.FONT_HERSHEY_COMPLEX,0.75,(255,255,0), 2)
    cv2.rectangle(image, (x,y),(x+w,y+h),(0,255,0),2)

    cv2.imshow("the face _recognised image is",image)
    cv2.waitKey(0)
cv2.destroyAllWindows()