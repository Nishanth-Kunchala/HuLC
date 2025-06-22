import serial
import time
import numpy as np

PORT = 'COM3'  # Replace with your Arduino port
BAUD = 9600

arduino = serial.Serial(PORT, BAUD, timeout=1)
time.sleep(2)

import numpy as np

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


# actuator_inv_kinematics([0, 0, 17, 90, 90, 45])
# Example: Sweep all 6 pins up and down together
try:
    ln = actuator_inv_kinematics([0, 5, 14, 0, 0, 0])
    print(ln)
    send_pwm_list(ln)
except KeyboardInterrupt:
    print("Stopped.")
