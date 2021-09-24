import cv2
import mediapipe as mp
import time

# Minimum required code to access HandTracking from MediaPipe


# Accessing webcam
cap = cv2.VideoCapture(0)

# Defining variables to draw handtracking
mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

# Defining variables to hold time for fps overlay
pTime = 0
cTime = 0

# While loop to run code
while True:

    # Checking if webcam is alive
    success, img = cap.read()
    # Converting webcam feed to RGB
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Drawing handtracking to RGB image
    results = hands.process(imgRGB)
    # print(results.multi_hand_landmarks)

    # Checking if a hand was drawn on webcamfeed
    if results.multi_hand_landmarks:
        # For-loop for every hand found in the webcamfeed
        for handLms in results.multi_hand_landmarks:
            # For-loop for every hand identified landmarks in every hand
            for id, lm in enumerate(handLms.landmark):
                # Converting the landmarks found on the hand to X and Y pixel coordinates and printing them
                # print(id,lm)
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                print(id, cx, cy)
                # Printing a big circle around landmark number 4 on the image
                if id == 4:
                    cv2.circle(img, (cx, cy), 25, (255, 0, 255), cv2.FILLED)
            # Drawing the landmarks and landmark connections from the hand into the webcamfeed
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

    # Calculating frames per second
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    # Drawing the frames per second to the webcamfeed
    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

    # Showing the image in a window
    cv2.imshow("Image", img)
    # A one millisecond delay
    cv2.waitKey(1)
