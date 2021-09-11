import time
import numpy as np

import cv2
import HandTrackingModule as htm
import mediapipe
import autopy

width_cam, height_cam = 640, 400
frame_reduction = 100
smoothening = 2

prev_time = 0
prev_locx, prev_locy = 0, 0
curr_locx, curr_locy = 0, 0

cap = cv2.VideoCapture(0)
cap.set(3, width_cam)
cap.set(4, height_cam)

detector = htm.handDetector(max_hands=1)
width_screen, height_screen = autopy.screen.size()
#print(width_screen, height_screen)

while True:
    success, img = cap.read()
    
    # find hand and finger landmarks
    img = detector.findHands(img)
    lm_list, bbox = detector.findPosition(img, draw=False)

    # Get the tip of the index and middle finger
    if len(lm_list) != 0:
        x1, y1 = lm_list[8][1:]
        x2, y2 = lm_list[12][1:]
        #print(x1, y1, x2, y2)

        # Check which finger are up
        fingers = detector.fingersUp()
        #print(fingers)
        cv2.rectangle(img, (frame_reduction, frame_reduction), 
                      (width_cam - frame_reduction, height_cam - frame_reduction),
                      (255, 0, 0), 2)
            
        # Only index finger up (moving mode)
        if fingers[1] == 1 and fingers[2] == 0:

            # Convert coordinates 
            x3 = np.interp(x1, (frame_reduction, width_cam - frame_reduction), (0, width_screen))
            y3 = np.interp(y1, (frame_reduction, height_cam - frame_reduction), (0, height_screen))

            # Smoothen the value
            curr_locx = prev_locx + (x3 - prev_locx) / smoothening
            curr_locy = prev_locy + (y3 - prev_locy) / smoothening

            # Moving mouse
            autopy.mouse.move(width_screen - curr_locx, curr_locy)
            cv2.circle(img, (x1, y1), 6, (255, 0, 0), cv2.FILLED)
            prev_locx, prev_locy = curr_locx, curr_locy

        # Both index and middle fingers up (clicking mode)
        if fingers[1] == 1 and fingers[2] == 1:
            length, img, line_info = detector.findDistance(8, 12, img)
            print(length)

            # Click motion
            if length < 25 and length >= 15:
                cv2.circle(img, (line_info[4], line_info[5]), 10, (0, 255, 0), cv2.FILLED)
                autopy.mouse.click()

    # show frame rate 
    curr_time = time.time()
    fps = 1/(curr_time - prev_time)
    prev_time = curr_time
    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (250, 0, 0), 3)

    cv2.imshow('image', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):      # press Q to quit
        break