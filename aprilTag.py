import cv2
import numpy as np
from robotpy_apriltag import AprilTagDetector
from wpimath.geometry import Translation2d

# Initialize webcam
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Initialize the AprilTag detector
detector = AprilTagDetector()
detector.addFamily("tag36h11")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to grayscale for tag detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Preprocessing: equalize and denoise
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    gray = cv2.equalizeHist(gray)

    # Detect tags
    detections = detector.detect(gray)

    centers = []
    for detection in detections:
        center = detection.getCenter()
        cx, cy = int(center.x), int(center.y)
        centers.append((cx, cy))

        # Draw center
        cv2.circle(frame, (cx, cy), 5, (255, 0, 0), -1)

        # Draw ID and corners
        tag_id = detection.getId()
        corner_buf = [0.0] * 8
        corners = detection.getCorners(corner_buf)
        corner_pts = [(int(corners[i]), int(corners[i+1])) for i in range(0, 8, 2)]

        # Draw lines between corners
        for i in range(4):
            pt1 = corner_pts[i]
            pt2 = corner_pts[(i + 1) % 4]
            cv2.line(frame, pt1, pt2, (0, 255, 255), 2)
        cv2.putText(frame, f"ID: {tag_id}", (cx + 10, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # If exactly 3 tags are detected, draw triangle and center
    if len(centers) == 3:
        center_x = int(sum([pt[0] for pt in centers]) / 3)
        center_y = int(sum([pt[1] for pt in centers]) / 3)

        cv2.line(frame, centers[0], centers[1], (0, 255, 0), 2)
        cv2.line(frame, centers[1], centers[2], (0, 255, 0), 2)
        cv2.line(frame, centers[2], centers[0], (0, 255, 0), 2)

        cv2.circle(frame, (center_x, center_y), 8, (0, 0, 255), -1)
        cv2.putText(frame, "Center", (center_x + 10, center_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    cv2.putText(frame, f"Tags detected: {len(detections)}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    # Show window
    cv2.imshow("AprilTag Tracker", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
