import pandas as pd
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from tqdm import tqdm

# --- CONFIG ---
input_folder = 'data'
output_folder = 'rotated_data'
csv_path = 'labeling.csv'
rotated_csv_path = 'labeling_rotated.csv'
angle_range = (-180, 180)
plot_n_examples = 5

os.makedirs(output_folder, exist_ok=True)

# --- LOAD CSV ---
df = pd.read_csv(csv_path)
print(f"Loaded {len(df)} entries.")

rotated_records = []

# --- ROTATE & ADJUST CENTERPOINT ---
for idx, row in tqdm(df.iterrows(), total=len(df)):
    x, y, filename = row['x'], row['y'], row['image_name']
    x, y = float(x), float(y)

    image_path = os.path.join(input_folder, filename)
    image = cv2.imread(image_path)

    if image is None:
        print(f"Skipping missing image: {image_path}")
        continue

    h, w = image.shape[:2]
    center = (w // 2, h // 2)

    angle = np.random.uniform(*angle_range)
    M = cv2.getRotationMatrix2D(center, angle, scale=1.0)

    # Rotate image
    rotated_image = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

    # Update point using same rotation matrix
    point = np.array([[x, y]], dtype=np.float32)
    new_point = cv2.transform(np.array([point]), M)[0][0]
    new_x, new_y = new_point

    # Save new image and record
    new_filename = f"rotated_{idx:04d}.png"
    cv2.imwrite(os.path.join(output_folder, new_filename), rotated_image)
    rotated_records.append({'x': new_x, 'y': new_y, 'filepath': new_filename})

# --- SAVE NEW CSV ---
rotated_df = pd.DataFrame(rotated_records)
rotated_df.to_csv(rotated_csv_path, index=False)
print(f"Saved rotated CSV to {rotated_csv_path}")

# --- VISUALIZE A FEW ---
for i in range(min(plot_n_examples, len(rotated_df))):
    row = rotated_df.iloc[i]
    image_path = os.path.join(output_folder, row['filepath'])
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    plt.imshow(image_rgb)
    plt.scatter(row['x'], row['y'], color='red', marker='x', s=50)
    plt.title(f"Rotated Image {i} with Keypoint")
    plt.axis('off')
    plt.show()
