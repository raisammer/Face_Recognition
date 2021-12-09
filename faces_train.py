# -*- coding: utf-8 -*-
"""
Created on Thu Nov 25 08:37:12 2021

@author: dell
"""

import os
import cv2
import numpy as np

people = ['alexander dadario' , 'cute girl' , 'shirleu sethia' ,'shruti' ,'tamana']

p =[]
for i in os.listdir(r'C:\Users\dell\Pictures\open cv'):
    p.append(i)
print (p)

paths =r'C:\Users\dell\Pictures\open cv'

haar_cascade = cv2.CascadeClassifier('C:/Users/dell/c,c++/.vscode/11.xml/haar_faces.xml')

#features are actually he image arrays  and the labels are the name corresponding to that list
features = []
labels =[]

def create_train():
    for person in people :
        path = os.path.join(paths,person)
        label = people.index(person)
        
        for image in os.listdir(path):
            image_path = os.path.join(path,image)
            
            image_array = cv2.imread(image_path)
            
            gray= cv2.cvtColor(image_array,cv2.COLOR_BGR2GRAY)
            
            face_rect = haar_cascade.detectMultiScale(gray , scaleFactor =1.1 , minNeighbors =5)
            
            for (x,y,w,h) in face_rect :
                face_rectangle = cv2.rectangle(image_array ,(x,y),(x+w,y+h), (0,255,0),2)
                faces_roi = gray[y:y+h , x:x+w]
                features.append(faces_roi)
                labels.append(label)
            cv2.imshow("onebyone",image_array)
            cv2.waitKey(0)
           # print(features)
           # print(labels)
            
create_train()
print("the training done------------------------")
features = np.array(features , object)
labels=np.array(labels)
print(len(labels)) 
print(len(features))
print(labels)
print(features[0][59][1])    
print(features[0])



face_recognizer = cv2.face.LBPHFaceRecognizer_create()

#train the recognizer on the features list and the labels list

face_recognizer.train(features,labels)
face_recognizer.save('face_traing.yml')
np.save('features.npy', features)
np.save('labels.npy',labels)

cv2.destroyAllWindows()
