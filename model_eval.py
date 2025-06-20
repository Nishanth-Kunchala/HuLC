import tensorflow as tf
import pandas as pd
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt

def heatmap_to_weighted_coords(heatmap_x, heatmap_y, top_k=10):
    """Extracts (x, y) coordinates by computing a weighted average of the top-k probabilities."""
    
    # Get the top-k indices based on probability values
    top_x_indices = np.argsort(heatmap_x)[-top_k:]  # Indices of top 10 probabilities for x
    top_y_indices = np.argsort(heatmap_y)[-top_k:]  # Indices of top 10 probabilities for y
    
    # Get corresponding probabilities
    top_x_probs = heatmap_x[top_x_indices]
    top_y_probs = heatmap_y[top_y_indices]
    
    # Normalize probabilities so they sum to 1 (convert them into weights)
    top_x_probs /= np.sum(top_x_probs)
    top_y_probs /= np.sum(top_y_probs)
    
    # Compute the weighted average of the top-k indices
    pred_x = np.sum(top_x_indices * top_x_probs)  # Weighted sum for x
    pred_y = np.sum(top_y_indices * top_y_probs)  # Weighted sum for y
    
    return pred_x, pred_y  # More stable keypoint prediction

def as_image(image_path):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, (img_height, img_width))
    image = tf.image.convert_image_dtype(image, tf.float32) / 255.0
    return image

img_height = img_width = 224
image_dir = 'data/'
model = tf.keras.models.load_model("xy_heatmap_model.keras")

for image_filename in os.listdir(image_dir):  # Test on a few images
    image_path = os.path.join(image_dir, image_filename)
    print(image_path)
    image = as_image(image_path)  # Load image
    image = tf.expand_dims(image, axis=0)  # Add batch dimension

    predictions = model.predict(image)  # Get model outputs

    # Convert heatmap outputs to (x, y) coordinates
    predicted_x, predicted_y = heatmap_to_weighted_coords(predictions[0][0], predictions[1][0])

    # Reshape heatmaps to match 512x512 image size
    heatmap_x = predictions[0][0].reshape(1, 512)  # Reshape X heatmap
    heatmap_y = predictions[1][0].reshape(512, 1)  # Reshape Y heatmap

    # Create combined 2D heatmap by outer product (for visualization)
    combined_heatmap = heatmap_x * heatmap_y  # Outer product creates a 2D probability map

    # Convert TensorFlow tensor to NumPy for visualization
    image_np = image.numpy()[0]  # Remove batch dimension

    # Plot image and heatmap side by side
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # Original image with predicted point
    axes[0].imshow(image_np)
    axes[0].scatter(predicted_x, predicted_y, color='yellow', marker='x', s=50, label="Predicted Keypoint")
    axes[0].set_title("Original Image with Prediction")
    axes[0].axis("off")

    # Heatmap visualization
    im = axes[1].imshow(combined_heatmap, cmap="inferno", extent=[0, 512, 0, 512])
    axes[1].set_title("Predicted Heatmap")
    axes[1].set_xlabel("X coordinate")
    axes[1].set_ylabel("Y coordinate")
    plt.colorbar(im, ax=axes[1])

    plt.tight_layout()
    plt.show()

    print(f"Predicted Keypoint: ({predicted_x:.2f}, {predicted_y:.2f}) for {image_path}")