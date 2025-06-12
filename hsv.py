import cv2
import numpy as np

def nothing(x):
    pass

# Load a sample frame from your video
cap = cv2.VideoCapture('Sample 1.mp4')
ret, frame = cap.read()
if not ret:
    print("❌ Couldn't read frame. Check file path or codec.")
    exit()
cap.release()

cv2.namedWindow("HSV Adjuster")
cv2.createTrackbar("LH", "HSV Adjuster", 0, 179, nothing)
cv2.createTrackbar("LS", "HSV Adjuster", 0, 255, nothing)
cv2.createTrackbar("LV", "HSV Adjuster", 0, 255, nothing)
cv2.createTrackbar("UH", "HSV Adjuster", 179, 179, nothing)
cv2.createTrackbar("US", "HSV Adjuster", 255, 255, nothing)
cv2.createTrackbar("UV", "HSV Adjuster", 255, 255, nothing)

while True:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Get current positions of trackbars
    lh = cv2.getTrackbarPos("LH", "HSV Adjuster")
    ls = cv2.getTrackbarPos("LS", "HSV Adjuster")
    lv = cv2.getTrackbarPos("LV", "HSV Adjuster")
    uh = cv2.getTrackbarPos("UH", "HSV Adjuster")
    us = cv2.getTrackbarPos("US", "HSV Adjuster")
    uv = cv2.getTrackbarPos("UV", "HSV Adjuster")

    lower_hsv = np.array([lh, ls, lv])
    upper_hsv = np.array([uh, us, uv])

    mask = cv2.inRange(hsv, lower_hsv, upper_hsv)
    result = cv2.bitwise_and(frame, frame, mask=mask)

    cv2.imshow("Original", frame)
    cv2.imshow("Mask", mask)
    cv2.imshow("Filtered", result)

    key = cv2.waitKey(1)
    if key == 27:  # Press ESC to break
        print("Lower HSV:", lower_hsv)
        print("Upper HSV:", upper_hsv)
        break

cv2.destroyAllWindows()
