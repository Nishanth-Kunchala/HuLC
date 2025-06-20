import cv2
import numpy as np
from robotpy_apriltag import AprilTagDetector, AprilTagPoseEstimator
from wpimath.geometry import Transform3d
from wpimath.geometry import Rotation3d

# Known tag positions relative to the reference tag (tag 0)
TAG_POSITIONS = {
    0: (0, 0, 0),       # Reference tag at origin
    1: (0.5, 0, 0),     # Example positions
    2: (0, 0.5, 0),     # Adjust these to your actual layout
}

print("Starting camera...")
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    raise IOError("Cannot open webcam")
print("Camera initialized.")

# Initialize detector
detector = AprilTagDetector()
detector.addFamily("tag36h11")
print("AprilTag detector initialized.")

# Initialize pose estimator
estimator_config = AprilTagPoseEstimator.Config(
    tagSize=0.004,  # Tag size in meters
    fx=500,        # Focal length x (must calibrate)
    fy=500,        # Focal length y (must calibrate)
    cx=320,        # Optical center x (usually width/2)
    cy=240         # Optical center y (usually height/2)
)
estimator = AprilTagPoseEstimator(estimator_config)

# Initialize CLAHE for contrast enhancement
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

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
    detected_tags = {}

    for det in detections:
        tag_id = det.getId()
        center = det.getCenter()
        cx, cy = int(center.x), int(center.y)

        # Get pose estimation
        try:
            pose = estimator.estimate(det)
            if pose is None:
                continue
        except Exception as e:
                continue

        
        # Get corners using the correct method
        corners = []
        for i in range(4):
            corner = det.getCorner(i)
            corners.append((int(corner.x), int(corner.y)))

        # --- DRAW DOT NORMAL TO EDGE (e.g., corner 0 to corner 1) ---
        # Convert corners to numpy for easier math
        p0 = np.array(corners[0], dtype=np.float32)
        p1 = np.array(corners[1], dtype=np.float32)
        center = np.array([cx, cy], dtype=np.float32)

        # Map tag IDs to which edge (start, end) to use
        TAG_EDGE_MAP = {
            0: (2, 3),  # tag 0 → edge from corner 0 to 1
            1: (2, 3),  # tag 1 → edge from corner 2 to 3
            2: (2, 1),  # tag 2 → edge from corner 1 to 2
            # Add more as needed
        }


        # Get which edge to use for this tag
        if tag_id in TAG_EDGE_MAP:
            start_idx, end_idx = TAG_EDGE_MAP[tag_id]
        else:
            start_idx, end_idx = 0, 1  # default edge if tag not in map

        p_start = np.array(corners[start_idx], dtype=np.float32)
        p_end = np.array(corners[end_idx], dtype=np.float32)
        edge = p_end - p_start


        # Normal vector (rotate 90 degrees)
        normal = np.array([-edge[1], edge[0]])

        # Normalize
        normal_unit = normal / np.linalg.norm(normal)

        # Use fixed pixel offset instead of estimating inches
        offset_distance_pixels = 75
        offset_point = center + normal_unit * offset_distance_pixels
        offset_point_int = tuple(np.round(offset_point).astype(int))

        # Check bounds before drawing
        h, w = frame.shape[:2]
        x, y = offset_point_int
        if 0 <= x < w and 0 <= y < h:
            cv2.circle(frame, offset_point_int, 5, (0, 0, 255), -1)
        else:
            print(f"Dot for tag {tag_id} off-screen at {offset_point_int}")


        # Draw the offset point
        cv2.circle(frame, offset_point_int, 5, (0, 0, 255), -1)  # red dot    
        
        # Store tag information
        detected_tags[tag_id] = {
            'pixel_center': (cx, cy),
            'corners': corners,
            'translation': pose.translation(),
            'rotation': pose.rotation()
        }

        # Visualize
        cv2.circle(frame, (cx, cy), 5, (255, 0, 0), -1)
        for j in range(4):
            cv2.line(frame, corners[j], corners[(j+1)%4], (0, 255, 0), 2)
        cv2.putText(frame, f"ID: {tag_id}", (cx + 10, cy), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        # Calculate relative positions if reference tag is found
        if 0 in detected_tags:
            ref_translation = detected_tags[0]['translation']
            
            for tag_id, tag_data in detected_tags.items():
                if tag_id == 0:
                    continue
                    
                if tag_id in TAG_POSITIONS:
                    known_pos = TAG_POSITIONS[tag_id]
                    # Display relative position information
                    # cv2.putText(frame, f"Rel Pos: {known_pos}", 
                    #          (tag_data['pixel_center'][0] + 10, tag_data['pixel_center'][1] + 20),
                    #          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

    cv2.imshow("AprilTag Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()