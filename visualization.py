import cv2
import numpy as np
import tensorflow as tf
import keras
import matplotlib.pyplot as plt
from matplotlib import cm

# Load your keypoint detection model
model = tf.keras.models.load_model(r'C:\Users\rajku\Documents\HuLC\HuLC\xy_heatmap_model.h5')

# Heatmap to weighted coordinates (same as yours)
def heatmap_to_weighted_coords(heatmap_x, heatmap_y, top_k=10):
    top_x_indices = np.argsort(heatmap_x)[-top_k:]
    top_y_indices = np.argsort(heatmap_y)[-top_k:]

    top_x_probs = heatmap_x[top_x_indices]
    top_y_probs = heatmap_y[top_y_indices]

    top_x_probs /= np.sum(top_x_probs)
    top_y_probs /= np.sum(top_y_probs)

    pred_x = np.sum(top_x_indices * top_x_probs)
    pred_y = np.sum(top_y_indices * top_y_probs)

    return pred_x, pred_y

# Preprocessing function (resize and normalize as needed)
def preprocess_frame(frame):
    frame_resized = cv2.resize(frame, (224, 224))  # resize to 224x224
    frame_normalized = frame_resized.astype(np.float32) / 255.0
    return np.expand_dims(frame_normalized, axis=0)

# Heatmap overlay function using matplotlib colormap
def overlay_heatmap_on_frame(frame, heatmap):
    heatmap_normalized = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)
    heatmap_colored = cm.inferno(heatmap_normalized.astype(np.uint8))[:, :, :3]  # RGB only
    heatmap_colored = (heatmap_colored * 255).astype(np.uint8)
    heatmap_bgr = cv2.cvtColor(heatmap_colored, cv2.COLOR_RGB2BGR)
    overlay = cv2.addWeighted(frame, 0.6, heatmap_bgr, 0.4, 0)
    return overlay

# Start video capture
print("Starting camera...")
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    raise IOError("Cannot open webcam")
print("Camera initialized.")

print("Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize frame for overlay display (retain original for visualization)
    display_frame = cv2.resize(frame, (512, 512))

    # Preprocess frame for model input
    input_frame = preprocess_frame(display_frame)

    # Predict using model
    predictions = model.predict(input_frame)

    heatmap_x = predictions[0][0].reshape(1, 512)
    heatmap_y = predictions[1][0].reshape(512, 1)

    # Compute full 2D heatmap
    combined_heatmap = heatmap_x * heatmap_y

    # Get keypoint prediction
    pred_x, pred_y = heatmap_to_weighted_coords(predictions[0][0], predictions[1][0])

    # After you get predicted keypoint at (pred_x, pred_y) in heatmap coords (224x224)
    scale_factor = 512 / 224
    display_x = int(pred_x * scale_factor)
    display_y = int(pred_y * scale_factor)


    # Overlay heatmap
    overlayed = overlay_heatmap_on_frame(display_frame, combined_heatmap)

    # Draw predicted keypoint on top
    cv2.drawMarker(overlayed, (int(display_x), int(display_y)), color=(0, 255, 255), markerType=cv2.MARKER_CROSS, markerSize=10, thickness=2)

    # Show output
    cv2.imshow("Keypoint Detection", overlayed)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()