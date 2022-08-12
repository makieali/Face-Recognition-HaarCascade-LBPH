

import cv2
import numpy as np
from PIL import Image #pillow package
import os

# Path for face image database
path = 'datasetsampleface'
path2 = 'datasetsampleeye'



# function to get the images and label data
def getImagesAndLabels(path):
    # get the path of all the files in the folder
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    # create empth face list
    faces = []
    # create empty ID list
    Ids = []
    # now looping through all the image paths and loading the Ids and the images
    for imagePath in imagePaths:
        # loading the image and converting it to gray scale
        pilImage = Image.open(imagePath).convert('L')
        # Now we are converting the PIL image into numpy array
        imageNp = np.array(pilImage, 'uint8')
        # getting the Id from the image
        ID = int(os.path.split(imagePath)[-1].split(".")[1])
        # extract the face from the training image sample
        faces.append(imageNp)
        Ids.append(ID)
    return faces, Ids
#####################################################################
def train():

    path = 'datasetsampleface'
    path2 = 'datasetsampleeye'

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml");
    detector2 = cv2.CascadeClassifier("haarcascade_eye_tree_eyeglasses.xml");

    print("\n [INFO] Training faces. It will take a few seconds. Wait ...")

    faces, ids = getImagesAndLabels(path)

    eyes, ids = getImagesAndLabels(path2)

    recognizer.train(faces, np.array(ids))
    recognizer.write('trainersample/trainer.yml')

    recognizer.train(eyes,np.array(ids))

    # Save the model into trainer/trainer.yml
    recognizer.write('trainersample/trainer2.yml')


#######################################################

train()