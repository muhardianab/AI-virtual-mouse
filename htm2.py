import cv2
import mediapipe as mp
import time

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

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        #print(results.multi_hand_landmarks)

        if self.results.multi_hand_landmarks:
            for hand_lms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, hand_lms, self.mpHands.HAND_CONNECTIONS)
        return img
    
    def findPosition(self, img, hand_number=0, draw=True):
        lm_list = []
        if self.results.multi_hand_landmarks:
            my_hand = self.results.multi_hand_landmarks[hand_number]

            for id, lm in enumerate(my_hand.landmark):
                #print(id, lm)
                height, width, channel = img.shape
                center_x, center_y =  int(lm.x*width), int(lm.y*height)
                #print(id, center_x, center_y)
                lm_list.append([id, center_x, center_y])
                if draw:
                    cv2.circle(img, (center_x, center_y), 10, (255, 0, 0), cv2.FILLED)
            
        return lm_list


def main():
    prev_time = 0
    curr_time = 0
    cap = cv2.VideoCapture(0)
    detector = handDetector()
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

if __name__ == "__main__":
    main()