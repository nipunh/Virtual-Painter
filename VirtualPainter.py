import cv2
import numpy as np
import time
import os

import HandTrackingMin as htm

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

cap = cv2.VideoCapture(1)
cap.set(3, 1280)
cap.set(4, 720)

detector = htm.HandDetector(detectionConfidence=0.85, tracCon = 0.75)

while True:
    success, img = cap.read()

    # to nullify mirror effect
    img = cv2.flip(img, 1)

    # Find the hand landmarks
    img = detector.find_hands(img)
    lmList = detector.find_position(img, draw=False)
    if len(lmList) != 0:
        # tip of index finger and middle finger, removing id using slicing
        x1, y1 = lmList[8][1:]
        x2, y2 = lmList[12][1:]

        # check which fingers are up
        fingers = detector.fingersUp()
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
    cv2.imshow("Image1", img1)
    # cv2.imshow("ImageCanvas", imgCanvas)

    cv2.waitKey(1)
