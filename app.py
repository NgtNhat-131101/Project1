import keras
import cv2
import numpy as np
from tensorflow.keras.utils import img_to_array
# import imutils
import processing_data
import os

alphabet=['a','b','c','d', 'unknown']
model = keras.models.load_model("my_handgesture.h5")

def classify(image):
    image = cv2.resize(image, (224, 224))
    #image = image.astype("float") / 255.0
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    proba = model.predict(image)
    idx = np.argmax(proba)
    return alphabet[idx]

cap = cv2.VideoCapture(0)
while 1:
    ret, img = cap.read()
    #image = cv2.imread('amer_sign2.png')
    #cv2.imshow("image", image)
    img = cv2.flip(img, 3)
    top, right, bottom, left = 75, 350, 300, 590
    roi = img[top:bottom, right:left]
    roi=cv2.flip(roi,1)
    gray = processing_data.extract_skin(roi)
    cv2.imshow('roi',gray)
    cv2.imshow('img', img)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('s'):
        alpha=classify(gray)
        print(alpha)
        cv2.rectangle(img, (left, top), (right, bottom), (0,255,0), 2)
        font=cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img,alpha,(0,130),font,5,(0,0,255),2)
        #cv2.resize(img,(1000,1000))
    if key==ord('q'):
        break;
cap.release()
cv2.destroyAllWindows()
