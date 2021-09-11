import time

import numpy as np
import math

import cv2
import mediapipe as mp
import HandTrackingModule as htm
#import htm2 as htm
import alsaaudio

width_cam, height_cam = 640, 400

cap = cv2.VideoCapture(0)
cap.set(3, width_cam)
cap.set(4, height_cam)
prev_time = 0

detector = htm.handDetector(detection_conf=0.85)

audio = alsaaudio.Mixer()
vol = 0
vol_bar = 0
vol_per = 0

while True:
    success, img = cap.read()
    img = detector.findHands(img)
    lm_list = detector.findPosition(img, draw=False)
    if len(lm_list) != 0:
        #print(lm_list[4], lm_list[8])

        x1, y1 = lm_list[4][1], lm_list[4][2]
        x2, y2 = lm_list[8][1], lm_list[8][2]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        cv2.circle(img, (x1, y1), 10, (255, 0, 50), cv2.FILLED)
        cv2.circle(img, (x2, y2), 10, (255, 0, 200), cv2.FILLED)
        cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 3)
        cv2.circle(img, (cx, cy), 10, (255, 50, 100), cv2.FILLED)

        length = math.hypot(x2-x1, y2-y1)
        #print(length)

        # Setting volume slide
        # Hand range 20-200 with distance +-30cm from camera
        # Volume range 0-100, overamplified not include
        vol = np.interp(length, [20, 200], [0, 100])
        vol_bar = np.interp(vol, [0, 100], [0, 200])
        vol_per = np.interp(vol, [0, 100], [0, 100])
        print(int(length), vol)
        audio.setvolume(int(vol))

        if length < 50:
            cv2.circle(img, (cx, cy), 10, (0, 255, 0), cv2.FILLED)
    
    # Show volume bar
    cv2.rectangle(img, (20, 300), (50, 300-int(vol_bar)), (0, 255, 0), cv2.FILLED)
    cv2.rectangle(img, (20, 300), (50, 100), (0, 0, 0), 3)
    cv2.putText(img, f'{int(vol_per)} %', (15, 330), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 0), 2)

    # Show frame rate 
    curr_time = time.time()
    fps = 1/(curr_time - prev_time)
    prev_time = curr_time
    cv2.putText(img, f'FPS: {int(fps)}', (10, 20), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 0, 0), 2)

    cv2.imshow("img", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):      # press Q to quit
        break
