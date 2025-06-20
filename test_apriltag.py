import cv2
import numpy as np
from robotpy_apriltag import AprilTagDetector
from wpimath.geometry import Transform3d

print("Starting camera...")
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    raise IOError("Cannot open webcam")
print("Camera initialized.")

# Initialize the AprilTag detector
detector = AprilTagDetector()
detector.addFamily("tag36h11")
print("AprilTag detector initialized.")

# Initialize CLAHE for contrast enhancement
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

# Tag size in meters
TAG_SIZE = 0.065  # Adjust based on actual tag size

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # --- RED BACKGROUND SUPPRESSION ---
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 100, 100])
    upper_red2 = np.array([180, 255, 255])
    red_mask = cv2.inRange(hsv, lower_red1, upper_red1) | cv2.inRange(hsv, lower_red2, upper_red2)

    # Standard grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray[red_mask > 0] = 255  # Treat red background as white

    # Enhance contrast
    gray = clahe.apply(gray)

    # Detect AprilTags
    detections = detector.detect(gray)

    for det in detections:
        center = det.getCenter()
        cx, cy = int(center.x), int(center.y)

        # Draw the center
        cv2.circle(frame, (cx, cy), 5, (255, 0, 0), -1)

        # Optional: Get corners and draw the tag outline
        corner_buf = [0.0] * 8
        det.getCorners(corner_buf)
        pts = [(int(corner_buf[i]), int(corner_buf[i+1])) for i in range(0, 8, 2)]
        for j in range(4):
            cv2.line(frame, pts[j], pts[(j+1)%4], (0, 255, 0), 2)

    cv2.imshow("AprilTag Tracker", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
