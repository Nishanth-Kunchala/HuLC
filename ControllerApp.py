import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import threading
import numpy as np
import time
import cv2
import serial

from robotpy_apriltag import AprilTagDetector, AprilTagPoseEstimator
from wpimath.geometry import Quaternion

# === existing code chunks here ===
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

def actuator_inv_kinematics(pose):
    """
    pose = [X (m), Y (m), Z (m), phiX (rad), phiY (rad), phiZ (rad)]
    Returns actuator lengths (in inches) relative to a 14-inch rest length.
    """
    r2d = 180 / np.pi
    # print(r2d)
    pose[3] *= 1/r2d
    pose[4] *= 1/r2d
    pose[5] *= 1/r2d
    # print(pose)
    pose = np.array(pose)
    r = pose[0:3]
    ln0 = np.ones(6) * 14  # initial actuator length in inches

    # Convert constants from degrees to radians
    bas = np.deg2rad(26.80431573)
    bal = np.deg2rad(93.19568427)
    aas = np.deg2rad(15.34569581)
    aal = np.deg2rad(104.65430419)
    atoboffset = -1*(np.deg2rad(60) - 1/2*bas - 1/2*aas)

    # Platform attachment points (anp)
    anp = np.array([
        [np.cos(atoboffset), np.sin(atoboffset), 0],
        [np.cos(atoboffset + aal), np.sin(atoboffset + aal), 0],
        [np.cos(atoboffset + aas + aal), np.sin(atoboffset + aas + aal), 0],
        [np.cos(atoboffset + 2 * aal + aas), np.sin(atoboffset + 2 * aal + aas), 0],
        [np.cos(atoboffset + 2 * aas + 2 * aal), np.sin(atoboffset + 2 * aas + 2 * aal), 0],
        [np.cos(atoboffset + 3 * aal + 2 * aas), np.sin(atoboffset + 3 * aal + 2 * aas), 0],
    ]) * (13.90955515 / 2)

    # Base attachment points (bn)
    bn = np.array([
        [1, 0, 0],
        [np.cos(bas), np.sin(bas), 0],
        [np.cos(bas + bal), np.sin(bas + bal), 0],
        [np.cos(2 * bas + bal), np.sin(2 * bas + bal), 0],
        [np.cos(2 * bas + 2 * bal), np.sin(2 * bas + 2 * bal), 0],
        [np.cos(3 * bas + 2 * bal), np.sin(3 * bas + 2 * bal), 0],
    ]) * (13.5 / 2)

    # Build full rotation matrix from pose angles (converted to degrees)
    phiX, phiY, phiZ = pose[3] * r2d, pose[4] * r2d, pose[5] * r2d

    cx, sx = np.cos(np.deg2rad(phiX)), np.sin(np.deg2rad(phiX))
    cy, sy = np.cos(np.deg2rad(phiY)), np.sin(np.deg2rad(phiY))
    cz, sz = np.cos(np.deg2rad(phiZ)), np.sin(np.deg2rad(phiZ))

    rotX = np.array([[1, 0, 0],
                     [0, cx, -sx],
                     [0, sx, cx]])

    rotY = np.array([[cy, 0, sy],
                     [0, 1, 0],
                     [-sy, 0, cy]])

    rotZ = np.array([[cz, -sz, 0],
                     [sz, cz, 0],
                     [0, 0, 1]])

    RotationMatrix = rotZ @ rotY @ rotX

    # Transform platform attachment points and compute actuator lengths
    transformed_anp = (RotationMatrix @ anp.T).T + r
    ln = np.linalg.norm(transformed_anp - bn, axis=1) - ln0
    conversion_factor = 25.4/200*255
    ln*=conversion_factor

    return ln

def send_pwm_list(pwm_values):
    assert len(pwm_values) == 6, "Send exactly 6 values"
    line = ','.join(str(int(v)) for v in pwm_values) + '\n'
    arduino.write(line.encode())
    time.sleep(1)
    if arduino.in_waiting > 0:
        print("Arduino:", arduino.readline().decode().strip())

# Globals
# running = False
# frame_img = None
# cap, detector, estimator, config, clahe = initialize_apriltag_system()
# PORT = 'COM3'  # Adjust as needed
# BAUD = 9600
# arduino = serial.Serial(PORT, BAUD, timeout=1)
# pose_history = []
# curr_pose = [0, 0, 14, 0, 0, 0]

def start_tracking():
    global running
    running = True
    threading.Thread(target=track_loop, daemon=True).start()

def stop_tracking():
    global running
    running = False

def track_loop():
    global frame_img, curr_pose, pose_history
    count = 0
    while running:
        count += 1
        x, y, z, x_deg, y_deg, z_deg, frame = get_coords_from_frame(cap, detector, estimator, config, clahe)

        if frame is not None:
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb_frame)
            frame_img = ImageTk.PhotoImage(img)
            video_label.config(image=frame_img)

        if count == 1:
            time.sleep(3)

        if x is not None:
            pose = [x, y, z, x_deg, y_deg, z_deg]
            pose_history.append(pose)
            if len(pose_history) > 10:
                pose_history.pop(0)

            if count % 10 == 0 and len(pose_history) == 10:
                avg_pose = np.mean(pose_history, axis=0)
                curr_pose[0] += -1 * avg_pose[0]
                curr_pose[1] += -1 * avg_pose[1]
                ln = actuator_inv_kinematics(curr_pose)

                update_pose_display(avg_pose, ln)
                send_pwm_list(ln)

        time.sleep(0.01)

def update_pose_display(pose, ln):
    pos_str = f"Position (in): X={pose[0]:.2f}, Y={pose[1]:.2f}, Z={pose[2]:.2f}"
    rot_str = f"Rotation (deg): X={pose[3]:.2f}, Y={pose[4]:.2f}, Z={pose[5]:.2f}"
    pwm_str = f"PWM: " + ', '.join(f"{v:.1f}" for v in ln)

    pose_label.config(text=pos_str + "\n" + rot_str)
    pwm_label.config(text=pwm_str)

# === GUI Setup ===
root = tk.Tk()
root.title("AprilTag Tracking Dashboard")
root.geometry("800x600")

# Video frame
video_label = tk.Label(root)
video_label.pack(pady=10)

# Pose info
pose_label = tk.Label(root, text="Pose will show here", font=("Arial", 10))
pose_label.pack()

# PWM info
pwm_label = tk.Label(root, text="PWM will show here", font=("Arial", 10))
pwm_label.pack()

# Controls
btn_frame = tk.Frame(root)
btn_frame.pack(pady=10)

start_btn = tk.Button(btn_frame, text="Start Tracking", command=start_tracking, bg="green", fg="white")
start_btn.grid(row=0, column=0, padx=10)

stop_btn = tk.Button(btn_frame, text="Stop Tracking", command=stop_tracking, bg="red", fg="white")
stop_btn.grid(row=0, column=1, padx=10)

def on_closing():
    global running
    running = False
    cap.release()
    cv2.destroyAllWindows()
    root.destroy()

root.protocol("WM_DELETE_WINDOW", on_closing)
root.mainloop()
