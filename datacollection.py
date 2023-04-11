import cv2
import time

cap = cv2.VideoCapture(0)
folder = 'Data/Unknown'
counter = 0
while True:
    frame, img = cap.read()
    img = cv2.flip(img, 3)
    top, right, bottom, left = 76, 366, 300, 590

    hand = img[top:bottom, right:left]
    hand = cv2.flip(hand, 1)
    key = cv2.waitKey(1) & 0xFF

    cv2.imshow('hand', hand)
    cv2.imshow('image', img)

    if key == ord('s'):
        counter += 1
        cv2.imwrite(f'{folder}/Image_{time.time()}..jpg', hand)
        print(counter)

    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()