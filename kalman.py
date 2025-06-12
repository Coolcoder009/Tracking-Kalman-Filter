import cv2
import numpy as np

kalman = cv2.KalmanFilter(4, 2)

kalman.transitionMatrix = np.array([[1, 0, 1, 0],
                                    [0, 1, 0, 1],
                                    [0, 0, 1, 0],
                                    [0, 0, 0, 1]], np.float32)

kalman.measurementMatrix = np.eye(2, 4, dtype=np.float32)

kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03  
kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.5

lower_hsv = np.array([0, 0, 200])
upper_hsv = np.array([179, 60, 255])

cap = cv2.VideoCapture('Sample.mp4')
# Get frame width and height from the original video
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Define codec and create VideoWriter object
out = cv2.VideoWriter('tracked_output.mp4',
                      cv2.VideoWriter_fourcc(*'mp4v'),  # Codec
                      fps,
                      (frame_width, frame_height))      # Frame size

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_hsv, upper_hsv)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    measurement = None

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if 200 < area < 2000:
            (x, y), radius = cv2.minEnclosingCircle(cnt)
            circle_area = np.pi * radius * radius
            circularity = area / circle_area if circle_area > 0 else 0

            if circularity > 0.7:
                measurement = np.array([[np.float32(x)], [np.float32(y)]])
                cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 0), 2)
                cv2.putText(frame, "Detected", (int(x)-10, int(y)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
                break

    predicted = kalman.predict()
    pred_x, pred_y = int(predicted[0]), int(predicted[1])

    if measurement is not None:
        kalman.correct(measurement)
        track_color = (0, 255, 255)  # Yellow if corrected
    else:
        track_color = (255, 0, 0)    # Blue if just predicted (no detection)

    # Draw prediction
    cv2.circle(frame, (pred_x, pred_y), 8, track_color, 2)
    cv2.putText(frame, "Kalman", (pred_x + 10, pred_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, track_color, 2)

    cv2.imshow("Kalman Ball Tracker", frame)
    out.write(frame)

    if cv2.waitKey(30) & 0xFF == 27:
        break

cap.release()
out.release()
cv2.destroyAllWindows()
