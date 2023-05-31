
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
def  extract_skin(frame, HSV_lower=(0,0,0), HSV_upper=(230, 230, 230), YCrCb_lower=(0,135,85), YCrCb_upper=(255,180,135), show=True):
    
    img_HSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    #skin color range for hsv color space 
    HSV_mask = cv2.inRange(img_HSV, HSV_lower, HSV_upper)
    HSV_mask = cv2.morphologyEx(HSV_mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))

    #converting from gbr to YCbCr color space
    img_YCrCb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
 
    YCrCb_mask = cv2.inRange(img_YCrCb, YCrCb_lower, YCrCb_upper) 
    YCrCb_mask = cv2.morphologyEx(YCrCb_mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))

    #merge skin detection (YCbCr and hsv)
    global_mask=cv2.bitwise_and(YCrCb_mask,HSV_mask)
    global_mask=cv2.medianBlur(global_mask,3)
    global_mask = cv2.morphologyEx(global_mask, cv2.MORPH_OPEN, np.ones((4,4), np.uint8))


    HSV_result = cv2.bitwise_not(HSV_mask)
    YCrCb_result = cv2.bitwise_not(YCrCb_mask)

    skin = cv2.bitwise_not(global_mask)
    return skin

    # if show:
        # cv2.imshow("HSV skin",HSV_result)
        # cv2.imshow("YCrCb skin",YCrCb_result)
        # cv2.imshow("Global skin",skin)


# image = cv2.imread('Data/val/B/Image_1681277795.298572..jpg')
# plt.imshow(image)
# my_img = extract_skin(image)
# print(my_img.shape)
# plt.imshow(my_img, cmap = 'gray')
# plt.show()
