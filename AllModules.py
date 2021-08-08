# Hand gesture recognition module
# MediaPipe by google, has various trained model,

# Hand tracking module has : 1) Palm Detection and 2) Hand Landmarks

# Hand landmarks module finds 21 points on hand

# Google manually annotated 30K hands

import cv2
import mediapipe as mp

# to keep track of frames
import time
import numpy as np
import os


class HandDetector:
    def __init__(self, mode=False, maxHands=1, detectionConfidence=0.5 , tracCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionConfidence = detectionConfidence
        self.tracCon = tracCon
        self.pTime = 0

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

        # ret, buffer = cv2.imencode('.jpg', img)
        # frame = buffer.tobytes()
       
        # yield(b'--frame\r\n'
        #            b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

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

    def virtualPainter(self):
        folderPath = "Headers"
        myList = os.listdir(folderPath)
        # print(myList)

        overLaylist = []

        # Import the images
        for imgPath in myList:
            image = cv2.imread(f'{folderPath}/{imgPath}')
            overLaylist.append(image)
            # print(len(overLaylist))

        # Global variables
        header = overLaylist[6]
        drawColor = (255, 0, 255)
        brushThickness = 15
        eraserThickness = 75
        xp, yp = 0, 0

        # draw canvas
        imgCanvas = np.zeros((720, 1280, 3), np.uint8)

        cap = cv2.VideoCapture(0)
        cap.set(3, 1280)
        cap.set(4, 720)

        # detector = HandDetector(detectionConfidence=0.85, tracCon = 0.75)

        while True:
            success, img = cap.read()

            # to nullify mirror effect
            img = cv2.flip(img, 1)

            # Find the hand landmarks
            img = self.find_hands(img)
            lmList = self.find_position(img, draw=False)
            if len(lmList) != 0:
                # tip of index finger and middle finger, removing id using slicing
                x1, y1 = lmList[8][1:]
                x2, y2 = lmList[12][1:]

                # check which fingers are up
                fingers = self.fingersUp()
                # print(fingers)
                # Draw with 1 finger select with 2 fingers
                # If selection mode - two fingers up then select no draw
                if fingers[1] and fingers[2]:
                    # print('selection mode')
                    xp, yp = 0, 0

                    # each pallet is 135 pixel
                    # each gap is 55 pixel
                    # checking for the click
                    if y1 < 125:
                        if 55 < x1 < 190:
                            header = overLaylist[1]
                            drawColor = (0, 0, 255)
                        elif 245 < x1 < 380:
                            header = overLaylist[2]
                            drawColor = (87, 87, 255)
                        elif 435 < x1 < 570:
                            header = overLaylist[3]
                            drawColor = (235, 23, 94)
                        elif 625 < x1 < 760:
                            header = overLaylist[4]
                            drawColor = (250, 60, 255)
                        elif 815 < x1 < 950:
                            header = overLaylist[5]
                            drawColor = (89, 222, 255)
                        elif 1052 < x1 < 1170:
                            header = overLaylist[0]
                            drawColor = (0, 0, 0)

                    cv2.rectangle(img, (x1, y1 - 25), (x2, y2 + 25), drawColor, cv2.FILLED)
                    # print(drawColor)

                # If drawing mode when index finger is up
                if fingers[1] and fingers[2] == False:
                    cv2.circle(img, (x1, y1), 25, drawColor, cv2.FILLED)
                    # print('draw mode')
                    if xp == 0 and yp == 0:
                        xp, yp = x1, y1

                    if drawColor == (0, 0, 0):
                        cv2.line(img, (xp, yp), (x1, y1), drawColor, eraserThickness)
                        cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, eraserThickness)
                    else:
                        cv2.line(img, (xp, yp), (x1, y1), drawColor, brushThickness)
                        cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, brushThickness)

                    xp, yp = x1, y1

            # convert the canvas to black and white
            imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
            _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
            imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
            img = cv2.bitwise_and(img, imgInv)
            img1 = cv2.bitwise_or(img, imgCanvas)



            # Setting the header image
            img[0:125, 0:1280] = header
            img = cv2.addWeighted(img, 0.5, imgCanvas, 0.5, 0)
            # cv2.imshow("Image", img)
            # cv2.imshow("Image1", img1)
            # cv2.imshow("ImageCanvas", imgCanvas)
            
            ret, buffer = cv2.imencode('.jpg', img)
            frame=buffer.tobytes()
       
            yield(b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            
            cv2.waitKey(1)

    def fingerCounter(self):
        # import images
        folderPath = "Images"
        myList = os.listdir(folderPath)
        # print(myList)
        overlayList = []

        for imPath in myList:
            image = cv2.imread(f'{folderPath}/{imPath}')
            # print(f'{folderPath}/{imPath}')
            overlayList.append((image))

        tipIds = [4, 8, 12, 16, 20]
        
        cap = cv2.VideoCapture(0)
        cap.set(3, 1280)
        cap.set(4, 720)

        while True:
            successs, img = cap.read()
            img = self.find_hands(img)
            lmList = self.find_position(img, draw=False)

            if len(lmList) != 0:
                fingers = []

                # thumb, x axis comparison
                if lmList[tipIds[0]][1] > lmList[tipIds[0] - 1][1]:
                    fingers.append(1)
                else:
                    fingers.append(0)

                # fingers, y axis comparison
                for id in range(1, 5):
                    if lmList[tipIds[id]][2] < lmList[tipIds[id] - 2][2]:
                            fingers.append(1)
                    else:
                            fingers.append(0)

                # print(fingers)
                total_fingers = fingers.count(1)
                # print(total_fingers)

                h, w, c = overlayList[total_fingers-1].shape
                #-1 of list takes last element
                img[0:h, 0:w] = overlayList[total_fingers-1]

                cv2.rectangle(img, (20, 255), (150, 400), (255, 0, 0), cv2.FILLED)
                cv2.putText(img, str(total_fingers), (35, 375), cv2.FONT_HERSHEY_SIMPLEX, 5, (255, 255, 255), 20)

            cTime = time.time()
            fps = 1/(cTime-self.pTime)
            pTime = cTime

            cv2.putText(img, f'FPS: {int(fps)}', (400, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
            # cv2.imshow("Image", img)

            
            ret, buffer = cv2.imencode('.jpg', img)
            frame=buffer.tobytes()
       
            yield(b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            
            cv2.waitKey(1)



        

