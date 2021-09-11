import time
import numpy as np
import math

import cv2
import mediapipe as mp

class handDetector():
    def __init__(self, mode=False, max_hands=2, detection_conf=0.5, track_conf=0.5):
        self.mode = mode
        self.max_hands = max_hands
        self.detection_conf = detection_conf
        self.track_conf = track_conf

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.max_hands, 
                                        self.detection_conf, self.track_conf)
        self.mpDraw = mp.solutions.drawing_utils
        self.tip_ids = [4, 8, 12, 16, 20]       # tip finger id

    # Find hand landmarks
    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        #print(results.multi_hand_landmarks)

        if self.results.multi_hand_landmarks:
            for hand_lms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, hand_lms, self.mpHands.HAND_CONNECTIONS)
        return img
    
    # Find finger landmarks
    def findPosition(self, img, hand_number=0, draw=True):
        x_list = []
        y_list = []
        bbox = []
        self.lm_list = []

        if self.results.multi_hand_landmarks:
            my_hand = self.results.multi_hand_landmarks[hand_number]

            for id, lm in enumerate(my_hand.landmark):
                #print(id, lm)
                height, width, c = img.shape
                cx, cy =  int(lm.x * width), int(lm.y * height)
                x_list.append(cx)
                y_list.append(cy)
                #print(id, cx, cy)
                self.lm_list.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 6, (255, 0, 0), cv2.FILLED)
        
        x_min, x_max = min(x_list, default=0), max(x_list, default=0)
        y_min, y_max = min(y_list, default=0), max(y_list, default=0)
        bbox = x_min, y_min, x_max, y_max

        if draw:
            cv2.rectangle(img, (x_min - 20, y_min - 20), (x_max + 20, y_max + 20), 
                          (0, 255, 0), 2)
        
        return self.lm_list, bbox

    # Find finger if its up
    def fingersUp(self):
        fingers = []

        # Thumb
        if self.lm_list[self.tip_ids[0]][1] > self.lm_list[self.tip_ids[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)

        # Fingers
        for id in range(1, 5):
            if self.lm_list[self.tip_ids[id]][2] < self.lm_list[self.tip_ids[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)

            # totalFingers = fingers.count(1)

        return fingers

    # Find distance and average of 2 fingers
    def findDistance(self, p1, p2, img, draw=True, r=7, t=3):
        x1, y1 = self.lm_list[p1][1:]
        x2, y2 = self.lm_list[p2][1:]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        if draw:
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), t)
            cv2.circle(img, (x1, y1), r, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), r, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (cx, cy), r, (0, 0, 255), cv2.FILLED)
            length = math.hypot(x2 - x1, y2 - y1)

        return length, img, [x1, y1, x2, y2, cx, cy]

def main():
    prev_time = 0
    curr_time = 0
    cap = cv2.VideoCapture(0)
    detector = handDetector()
    while True:
        success, img = cap.read()
        img = detector.findHands(img)
        lm_list, bbox = detector.findPosition(img)
        if len(lm_list) != 0:
            print(lm_list[4])

        curr_time = time.time()
        fps = 1/(curr_time - prev_time)
        prev_time = curr_time
        
        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, 
                    (250, 0, 0), 3)

        cv2.imshow('image', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):      # press Q to quit
            break

if __name__ == "__main__":
    main()