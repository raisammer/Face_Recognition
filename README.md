# Face_Recognition

Here we have done the face recognition by first training our program by giving certain images
then reading the features of images and
recognising the faces with certain probability(or we can say predicting)*/


/***Firstly we will create the folder
that folder will contain the folder with the names of the person
Inside each folder images should be there of that person respecteively
100 images should be there  of each person so that ur predicition will be somewhat accurate/**


face_train :
Here we are first training our program so that it can store data in the form of features
so we will pass the path as folder path
 and then we will read each images by going inside each folder 
 and then saving the features of the images corresponding to each person
 and we are saving the features of the images by
 first detecting the face of the image and saving the region of interest into the features array
 and thats how we are training our program
 
code run:
we will pass an image 
and the image will be one of the images which we have train
then we will again use haar_face file to detect face 
and then we will collect the features in the array
 and then we will compare and predict the features with the trained features 
then the most closestresemblance coresponding feature index will pass 
 and the that index will be called 
 and the name corresponding to that index will most probab;y be the person


////** thsi will be our code for face recognition**/
