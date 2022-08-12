
import cv2
import numpy as np
import os

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainersample/trainer.yml')  # load trained model

recognizer2 = cv2.face.LBPHFaceRecognizer_create()
recognizer2.read('trainersample/trainer2.yml')
cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml");
eyeCascade = cv2.CascadeClassifier("haarcascade_eye_tree_eyeglasses.xml");

font = cv2.FONT_HERSHEY_SIMPLEX

# iniciate id counter, the number of persons you want to include
id = 1  # two persons (e.g. Jacob, Jack)

names = ['', 'Ali','Ali2']  # key in names, start from the second place, leave first empty

# Initialize and start realtime video capture
cam = cv2.VideoCapture(0)
cam.set(3, 640)  # set video widht
cam.set(4, 480)  # set video height

# Define min window size to be recognized as a face
minW = 0.1 * cam.get(3)
minH = 0.1 * cam.get(4)

while True:

    ret, img = cam.read()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(int(minW), int(minH)),
    )


    for (x, y, w, h) in faces:

        roi_gray = gray[y:y + h, x:x + w]
        roi_color = img[y:y + h, x:x + w]
        eye = eyeCascade.detectMultiScale(roi_gray)
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        id, confidence = recognizer.predict(gray[y:y + h, x:x + w])
        for (ex, ey, ew, eh) in eye:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (255, 0, 0), 2)
            id, coneye = recognizer2.predict(roi_gray[ey:ey+h,ex:ex+w])
            if (coneye < 50):
                #id = names[id]
                coneye = "  {0}%".format(round(100 - coneye))
            else:
                #id = "unknown"
                coneye = "  {0}%".format(round(100 - coneye))

            #cv2.putText(img, str(id), (ex + 5, ey - 5), font, 0.3, (255, 255, 255), 2)
            cv2.putText(img, str(coneye), (ex + 5, ey + eh - 5), font, 0.3, (255, 255, 0), 1)
        # Check if confidence is less them 100 ==> "0" is perfect match
        if (confidence < 50):
            #id = names[id]
            confidence = "  {0}%".format(round(100 - confidence))
        else:
            #id = "unknown"
            confidence = "  {0}%".format(round(100 - confidence))

        #cv2.putText(img, str(id), (x + 5, y - 5), font, 1, (255, 255, 255), 2)
        cv2.putText(img, str(confidence), (x + 5, y + h - 5), font, 1, (255, 255, 0), 1)

    cv2.imshow('camera', img)

    k = cv2.waitKey(100) & 0xff  # Press 'ESC' for exiting video
    if k == 27:
        break

# Do a bit of cleanup
print("\n [INFO] Exiting Program and cleanup stuff")
cam.release()
cv2.destroyAllWindows()