
import numpy as np
import cv2
import matplotlib.pyplot as plt
minValue = 70
def func(image):
    # frame = cv2.imread(image)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray ,(5 ,5) ,2)

    th3 = cv2.adaptiveThreshold(blur ,255 ,cv2.ADAPTIVE_THRESH_GAUSSIAN_C ,cv2.THRESH_BINARY_INV ,11 ,2)
    ret, res = cv2.threshold(th3, minValue, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    return res

# image = cv2.imread('Data/A/Image_1681134736.1865165..jpg')
# my_img = func(image)
# print(my_img.shape)
# plt.imshow(my_img, cmap = 'gray')
# plt.show()
