import socket
import json
import cv2
import numpy as np
from robotpy_apriltag import AprilTagDetector, AprilTagPoseEstimator
from wpimath.geometry import Quaternion

# Setup global detector and estimator (only once)
def initialize_apriltag_system():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise IOError("Cannot open webcam")
    else:
        print("camera initialized!")

    detector = AprilTagDetector()
    detector.addFamily("tag36h11")

    estimator_config = AprilTagPoseEstimator.Config(
        tagSize=0.04,
        fx=500,
        fy=500,
        cx=320,
        cy=240
    )
    estimator = AprilTagPoseEstimator(estimator_config)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    return cap, detector, estimator, estimator_config, clahe

# Extract x, y from 3D offset point in camera frame and draw visuals
def get_coords_from_frame(cap, detector, estimator, estimator_config, clahe, target_id=None):
    ret, frame = cap.read()
    if not ret:
        return None, None, None, None, None, None, frame

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 100, 100])
    upper_red2 = np.array([180, 255, 255])
    red_mask = cv2.inRange(hsv, lower_red1, upper_red1) | cv2.inRange(hsv, lower_red2, upper_red2)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray[red_mask > 0] = 255
    gray = clahe.apply(gray)

    detections = detector.detect(gray)

    for det in detections:
        tag_id = det.getId()
        if target_id is not None and tag_id != target_id:
            continue

        pose = estimator.estimate(det)
        if pose is None:
            continue

        # Draw center point and edges
        center = det.getCenter()
        cx, cy = int(center.x), int(center.y)
        cv2.circle(frame, (cx, cy), 5, (255, 0, 0), -1)
        for i in range(4):
            pt1 = det.getCorner(i)
            pt2 = det.getCorner((i+1)%4)
            cv2.line(frame, (int(pt1.x), int(pt1.y)), (int(pt2.x), int(pt2.y)), (0, 255, 0), 2)
        cv2.putText(frame, f"ID: {tag_id}", (cx + 10, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        TAG_EDGE_MAP = {0: (3, 2), 1: (3, 2), 2: (3, 2)}
        start_idx, end_idx = TAG_EDGE_MAP.get(tag_id, (0, 1))

        edge_vectors_local = {
            (0, 1): np.array([1, 0, 0]), (1, 2): np.array([0, -1, 0]),
            (2, 3): np.array([-1, 0, 0]), (3, 0): np.array([0, 1, 0]),
            (1, 0): np.array([-1, 0, 0]), (2, 1): np.array([0, 1, 0]),
            (3, 2): np.array([1, 0, 0]), (0, 3): np.array([0, -1, 0]),
        }
        local_edge = edge_vectors_local.get((start_idx, end_idx), np.array([1, 0, 0]))
        local_normal = np.cross(local_edge, np.array([0, 0, 1]))

        quat = pose.rotation().getQuaternion()
        qx, qy, qz, qw = quat.X(), quat.Y(), quat.Z(), quat.W()
        rotation_mat = np.array([
            [1 - 2*(qy**2 + qz**2), 2*(qx*qy - qz*qw),     2*(qx*qz + qy*qw)],
            [2*(qx*qy + qz*qw),     1 - 2*(qx**2 + qz**2), 2*(qy*qz - qx*qw)],
            [2*(qx*qz - qy*qw),     2*(qy*qz + qx*qw),     1 - 2*(qx**2 + qy**2)]
        ])

        def rotmat_to_euler(R):
            if abs(R[2, 0]) < 1.0:
                y_deg = -np.arcsin(R[2, 0])
                x_deg = np.arctan2(R[2, 1], R[2, 2])
                z_deg = np.arctan2(R[1, 0], R[0, 0])

            return np.rad2deg(x_deg), np.rad2deg(y_deg), np.rad2deg(z_deg)  # in radians
        
        x_deg, y_deg, z_deg = rotmat_to_euler(rotation_mat)

        translation = np.array([
            pose.translation().x,
            pose.translation().y,
            pose.translation().z
        ])

        normal_cam = rotation_mat @ local_normal
        offset_meters = 0.0254 * 3.25
        offset_point_cam = translation + normal_cam * offset_meters

        # Project back to image
        x, y, z = offset_point_cam
        fx, fy, cx_intr, cy_intr = estimator_config.fx, estimator_config.fy, estimator_config.cx, estimator_config.cy
        if z != 0:
            u = int(fx * x / z + cx_intr)
            v = int(fy * y / z + cy_intr)
            h, w = frame.shape[:2]
            if 0 <= u < w and 0 <= v < h:
                cv2.circle(frame, (u, v), 5, (255, 0, 255), -1)  # magenta = projected 3D point

        return float(x)*39.37, float(y)*39.37, float(z)*39.37, x_deg, y_deg, z_deg, frame

    return None, None, None, None, None, None, frame  # no tag found

#######################
### BEGIN MAIN CODE ###
#######################
# Initialize
cap, detector, estimator, config, clahe = initialize_apriltag_system()

# Setup UDP socket
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
server_address = ('localhost', 5005)

# Loop
count = 0
pose_history = []

print("starting tracking...")
while True:
    count += 1
    x, y, z, x_deg, y_deg, z_deg, frame = get_coords_from_frame(cap, detector, estimator, config, clahe)

    if x is not None and y is not None and z is not None:
        pose = [x, y, z, x_deg, y_deg, z_deg]
        pose_history.append(pose)
        
        # Keep only the last 20
        if len(pose_history) > 20:
            pose_history.pop(0)
        
        # Send over UDP
        # message = (json.dumps(pose) + '\n').encode('utf-8')
        # sock.sendto(message, server_address)
        # print(f"Sent coords: {pose}")

        # Every 20th iteration, print average
        if count % 20 == 0 and len(pose_history) == 20:
            avg_pose = np.mean(pose_history, axis=0)
            print(f"\nAverage of last 20 poses:")
            print(f"Position (in): X={avg_pose[0]:.2f}, Y={avg_pose[1]:.2f}, Z={avg_pose[2]:.2f}")
            print(f"Orientation (deg): X Rotation={avg_pose[3]:.2f}, Y Rotation={avg_pose[4]:.2f}, z Rotation={avg_pose[5]:.2f}\n")
    else:
        # print("No tag detected.")
        0

    cv2.imshow("AprilTag Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
sock.close()
cv2.destroyAllWindows()
