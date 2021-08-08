import cv2
import time
import os

import HandTrackingMin as htm

wCam, hCam = 1280, 720
pTime = 0

cap = cv2.VideoCapture(1)
# 3 for width, 4 for height
cap.set(3, wCam)
cap.set(4, hCam)

# import images
folderPath = "Images"
myList = os.listdir(folderPath)
# print(myList)
overlayList = []

for imPath in myList:
     image = cv2.imread(f'{folderPath}/{imPath}')
     # print(f'{folderPath}/{imPath}')
     overlayList.append((image))

print(len(overlayList))

detector = htm.HandDetector(detectionConfidence=0.75)

tipIds = [4, 8, 12, 16, 20]

while True:
     successs, img = cap.read()
     img = detector.find_hands(img)
     lmList = detector.find_position(img, draw=False)

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
          print(total_fingers)

          h, w, c = overlayList[total_fingers-1].shape
          #-1 of list takes last element
          img[0:h, 0:w] = overlayList[total_fingers-1]

          cv2.rectangle(img, (20, 255), (150, 400), (255, 0, 0), cv2.FILLED)
          cv2.putText(img, str(total_fingers), (35, 375), cv2.FONT_HERSHEY_SIMPLEX, 5, (255, 255, 255), 20)

     cTime = time.time()
     fps = 1/(cTime-pTime)
     pTime = cTime

     cv2.putText(img, f'FPS: {int(fps)}', (400, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
     cv2.imshow("Image", img)
     cv2.waitKey(1)



