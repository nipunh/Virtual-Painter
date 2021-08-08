# Hand gesture recognition module
# MediaPipe by google, has various trained model,

# Hand tracking module has : 1) Palm Detection and 2) Hand Landmarks

# Hand landmarks module finds 21 points on hand

# Google manually annotated 30K hands

import cv2
import mediapipe as mp

# to keep track of frames
import time

# setting camera for input
cap = cv2.VideoCapture(1)

# starting media pipe package
mpHands = mp.solutions.hands

# Initial parameters for Hands()
# static_image_mode=False,
# max_num_hands=2,
# min_detection_confidence=0.5,
# min_tracking_confidence=0.5
hands = mpHands.Hands()

# Draw the points on live video hands
mpDraw = mp.solutions.drawing_utils

# Global variables
prevTime = 0

while True:
    # //read video from source
    success, img = cap.read()

    # Converting image to RGB for model
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # This function will give points of hands in the video
    results = hands.process(imgRGB)

    # Detects hands and print points
    # print(results.multi_hand_landmarks)

    if results.multi_hand_landmarks:
        # print(results.multi_hand_landmarks)
        for handLms in results.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark):
                # we'll get 0 to 20 ids for each point and landmark will give x,y,z co-ordinated i.e. actually ratio
                # of image, we'll multiple with and height to get pixel value
                # print(id, lm)

                # Dimensions of image
                height, width, channel = img.shape
                cx, cy = int(lm.x * width), int(lm.y * height)

                # Draw circle of particular position
                # print(id, cx, cy)
                if id == 4:
                    cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)

            # draw points
            mpDraw.draw_landmarks(img, handLms)
            # Draw connections
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

    currentTime = time.time()
    fps = 1/(currentTime - prevTime)
    prevTime = currentTime

    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 255), 3)

    # show video using opencv
    cv2.imshow("Image", img)
    cv2.waitKey(1)
