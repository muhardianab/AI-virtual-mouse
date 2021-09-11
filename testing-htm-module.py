import time

import cv2
import mediapipe as mp
import HandTrackingModule as htm
    
prev_time = 0
curr_time = 0
cap = cv2.VideoCapture(0)
detector = htm.handDetector()

while True:
    success, img = cap.read()
    img = detector.findHands(img)
    lm_list = detector.findPosition(img)
    if len(lm_list) != 0:
        print(lm_list[4])

    curr_time = time.time()
    fps = 1/(curr_time - prev_time)
    prev_time = curr_time
    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (250, 0, 0), 3)

    cv2.imshow('image', img)
    cv2.waitKey(1)