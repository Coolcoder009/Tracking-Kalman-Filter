import cv2
import numpy as np

# Load your video
cap = cv2.VideoCapture('Sample 3.mp4')  # Replace with your actual file path

# Define white ball HSV range (low saturation, high brightness)
lower_hsv = np.array([0, 0, 200])
upper_hsv = np.array([179, 60, 255])

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Resize (optional)
    # frame = cv2.resize(frame, (640, 360))

    # Convert to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_hsv, upper_hsv)

    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = cv2.contourArea(cnt)

        # Ignore very small and very large areas
        if 200 < area < 2000:
            (x, y), radius = cv2.minEnclosingCircle(cnt)
            circle_area = np.pi * radius * radius
            circularity = area / circle_area if circle_area > 0 else 0

            # Keep only fairly circular objects
            if circularity > 0.7:
                center = (int(x), int(y))
                cv2.circle(frame, center, int(radius), (0, 255, 0), 2)      # Ball boundary
                cv2.circle(frame, center, 3, (0, 0, 255), -1)               # Center point
                cv2.putText(frame, f"Ball", (int(x) - 10, int(y) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # Show original and mask (for debugging)
    cv2.imshow("Ball Detection", frame)
    #cv2.imshow("White Mask", mask)

    if cv2.waitKey(30) & 0xFF == 27:  # Press ESC to quit
        break

cap.release()
cv2.destroyAllWindows()
