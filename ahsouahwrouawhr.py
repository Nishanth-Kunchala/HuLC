import cv2
import numpy as np
from robotpy_apriltag import AprilTagDetector, AprilTagPoseEstimator
from wpimath.geometry import Transform3d, Rotation3d, Quaternion
from wpimath.units import radians

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
# Update your estimator config with proper values for 1080p camera
estimator_config = AprilTagPoseEstimator.Config(
    tagSize=0.04,  # Tag size in meters
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

        # Estimate pose
        try:
            pose = estimator.estimate(det)
            if pose is None:
                continue
        except Exception as e:
            print(f"Pose estimation failed for tag {tag_id}: {e}")
            continue

        # Get corners
        corners = []
        for i in range(4):
            corner = det.getCorner(i)
            corners.append((int(corner.x), int(corner.y)))

        # Edge mapping per tag
        TAG_EDGE_MAP = {
            0: (2, 3),
            1: (2, 3),
            2: (2, 1),
        }

        # Choose edge
        if tag_id in TAG_EDGE_MAP:
            start_idx, end_idx = TAG_EDGE_MAP[tag_id]
        else:
            start_idx, end_idx = 0, 1  # default

        # Edge in tag-local frame (used for 3D offset)
        edge_vectors_local = {
            (0, 1): np.array([1, 0, 0]),
            (1, 2): np.array([0, -1, 0]),
            (2, 3): np.array([-1, 0, 0]),
            (3, 0): np.array([0, 1, 0]),
            (1, 0): np.array([-1, 0, 0]),
            (2, 1): np.array([0, 1, 0]),
            (3, 2): np.array([1, 0, 0]),
            (0, 3): np.array([0, -1, 0]),
        }

        local_edge = edge_vectors_local.get((start_idx, end_idx), np.array([1, 0, 0]))

        # Get normal vector in tag-local frame (90° CCW)
        local_normal = np.cross(local_edge, np.array([0, 0, 1]))  # Z is out of tag face

        # Get the rotation from the pose
        rotation = pose.rotation()
        
        # Get the rotation matrix using the toRotationMatrix() method
        rotation_mat = np.array([
            [rotation.X(), rotation.Y(), rotation.Z()],
            [0, 0, 0],  # Placeholder - need proper matrix conversion
            [0, 0, 0]   # Placeholder - need proper matrix conversion
        ])
        
        # Alternative approach using quaternion
        quat = rotation.getQuaternion()
        # Convert quaternion to rotation matrix (simplified example)
        # This is a placeholder - you'll need to implement proper quaternion to matrix conversion
        rotation_mat = np.eye(3)  # Identity matrix as placeholder

        translation = np.array([
            pose.translation().x,
            pose.translation().y,
            pose.translation().z
        ])
        
        # Transform normal to camera frame
        normal_cam = rotation_mat @ local_normal
        offset_meters = 0.0254 * 3  # 3 inches
        offset_point_cam = translation + normal_cam * offset_meters

        # Intrinsics (adjust to your camera)
        fx = estimator_config.fx
        fy = estimator_config.fy
        cx_intr = estimator_config.cx
        cy_intr = estimator_config.cy

        # Project 3D point to 2D
        x, y, z = offset_point_cam
        if z != 0:
            u = int(fx * x / z + cx_intr)
            v = int(fy * y / z + cy_intr)

            h, w = frame.shape[:2]
            if 0 <= u < w and 0 <= v < h:
                cv2.circle(frame, (u, v), 5, (255, 0, 255), -1)  # magenta 3D-projected dot
            else:
                print(f"Projected 3D point for tag {tag_id} is off-screen: {(u, v)}")

        # Store tag info
        detected_tags[tag_id] = {
            'pixel_center': (cx, cy),
            'corners': corners,
            'translation': translation,
            'rotation': rotation_mat
        }

        # Visualization
        cv2.circle(frame, (cx, cy), 5, (255, 0, 0), -1)
        for j in range(4):
            cv2.line(frame, corners[j], corners[(j+1)%4], (0, 255, 0), 2)
        cv2.putText(frame, f"ID: {tag_id}", (cx + 10, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

    cv2.imshow("AprilTag Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()