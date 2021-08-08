# Hand gesture recognition module
# MediaPipe by google, has various trained model,

# Hand tracking module has : 1) Palm Detection and 2) Hand Landmarks

# Hand landmarks module finds 21 points on hand

# Google manually annotated 30K hands

import cv2
import mediapipe as mp

# to keep track of frames
import time


class HandDetector:
    def __init__(self, mode=False, maxHands=1, detectionConfidence=0.5 , tracCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionConfidence = detectionConfidence
        self.tracCon = tracCon

        # starting media pipe package
        self.mpHands = mp.solutions.hands

        # Initial parameters for Hands()
        # static_image_mode=False,
        # max_num_hands=2,
        # min_detection_confidence=0.5,
        # min_tracking_confidence=0.5
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.detectionConfidence,  self.tracCon)

        # Draw the points on live video hands
        self.mpDraw = mp.solutions.drawing_utils

        self.tipIds = [4, 8, 12, 16, 20]



    def find_hands(self, img, draw = True):
        # Detects hands and print points
        # print(results.multi_hand_landmarks)

        self.imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # This function will give points of hands in the video
        self.results = self.hands.process(self.imgRGB)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    # draw lines
                    self.mpDraw.draw_landmarks(img, handLms)
                    # Draw connections
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
                    # we'll get 0 to 20 ids for each point and landmark will give x,y,z co-ordinated i.e. actually ratio
                    # of image, we'll multiple with and height to get pixel value
                    # print(id, lm)

        return img

    def find_position(self, img, hand_no=0, draw=True):
        self.land_mark_list = []

        if self.results.multi_hand_landmarks:
            my_hand = self.results.multi_hand_landmarks[hand_no]
            for id, lm in enumerate(my_hand.landmark):
                # Dimensions of image
                height, width, channel = img.shape
                cx, cy = int(lm.x * width), int(lm.y * height)

                # Draw circle of particular position
                # print(id, cx, cy)
                self.land_mark_list.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

        return self.land_mark_list

    def fingersUp(self):
        fingers = []

        # thumb, x axis comparison
        if self.land_mark_list[self.tipIds[0]][1] < self.land_mark_list[self.tipIds[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)

        # fingers, y axis comparison
        for id in range(1, 5):
            if self.land_mark_list[self.tipIds[id]][2] < self.land_mark_list[self.tipIds[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
        return fingers


if __name__ == "__main__":

    # Global variables
    prevTime = 0
    currentTime = 0

    # setting camera for input
    cap = cv2.VideoCapture(1)

    detector = HandDetector()

    while True:
        # //read video from source
        success, img = cap.read()

        img = detector.find_hands(img)

        lmList = detector.find_position(img)

        # if len(lmList) != 0:
        #     print(lmList)

        # Calculating FPS
        currentTime = time.time()
        fps = 1 / (currentTime - prevTime)
        prevTime = currentTime

        # Adding FPS to image
        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 255), 3)

        # show video using opencv
        cv2.imshow("Image", img)
        cv2.waitKey(1)
