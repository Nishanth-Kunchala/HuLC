import cv2
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Try different backends
if not cap.isOpened():
    print("Error: Could not open camera.")
else:
    print("Camera opened successfully!")
    cap.release()