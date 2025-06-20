import os
import random
import json
import pickle

import tensorflow as tf
import kagglehub as kh
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy

os.environ["CUDA_VISIBLE_DEVICES"] = "0"    # -1 --> CPU and 0 --> GPU 

gpus = tf.config.list_physical_devices('GPU')

if gpus: 
    tf.config.set_logical_device_configuration(
        gpus[0],
        [tf.config.LogicalDeviceConfiguration(memory_limit=14000)] # Memory limit is in MB here --> 14 GB of ram | CHANGE BASED ON SYSTEM SPECS
    )

logical_gpus = tf.config.list_logical_devices('GPU')
print(len(gpus), "Physical GPU,", len(logical_gpus), "Logical GPUs")

tf.random.set_seed(42)

path = kh.dataset_download("msafi04/iss-docking-dataset")
print(path)

df = pd.read_csv(os.path.join(path, 'train.csv'))
df.head()

n_imgs = len(df)
n_imgs

plt.hist(df['distance'], bins=30)
plt.xlabel('Distance Value')
plt.ylabel('Count')
plt.title('Distance Value Histogram')
plt.show()

locations = tuple(map(lambda x: tuple(json.loads(x)), df['location']))
locations = np.array(locations)
x, y = locations[:,0], locations[:, 1]

plt.hist2d(x,y, bins=[np.arange(0, 512, 12),np.arange(0, 512, 12)])
plt.colorbar()
plt.show()

img_width = img_height = 512
batch_size = 32

x = np.arange(0, img_width, 1)
y = np.arange(0, img_height, 1)

x, y = np.meshgrid(x, y)
z = np.zeros(x.shape)

for _ in df['location']:
    coord = tuple(json.loads(_))
    z[coord[0]][coord[1]] += 1

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
surf = ax.plot_surface(x, y, z, cmap='viridis',
                       linewidth=0, antialiased=False)
fig.colorbar(surf)
plt.show()

del x, y, z, locations

image_paths = []
for f in os.listdir(os.path.join(path, 'train')):
    if f.startswith('.') or '.jpg' not in f:
        continue
    image_paths.append(os.path.join(path, 'train', f))
len(image_paths)

locations = df.get('location').tolist()

image_paths.sort(key=lambda x: int(''.join(filter(str.isdigit, x))))

combined_locations = list(zip(image_paths, locations))
random.Random(42).shuffle(combined_locations)

image_paths_locations, locations = zip(*combined_locations)

train_split = int(0.7*n_imgs)
val_split = train_split + int(0.1*n_imgs)
test_split = val_split + int(0.2*n_imgs)
print(train_split, val_split, test_split)

train_image_paths_locations = np.array(image_paths_locations[:train_split]).flatten()
train_locations = locations[:train_split]

val_image_paths_locations = np.array(image_paths_locations[train_split:val_split]).flatten()
val_locations = locations[train_split:val_split]

test_image_paths_locations = np.array(image_paths_locations[val_split:test_split]).flatten()
test_locations = locations[val_split:test_split]

train_locations = tuple(map(lambda x: tuple(json.loads(x)), train_locations))
train_locations = np.array(train_locations)/512

val_locations = tuple(map(lambda x: tuple(json.loads(x)), val_locations))
val_locations = np.array(val_locations)/512

test_locations = tuple(map(lambda x: tuple(json.loads(x)), test_locations))
test_locations = np.array(test_locations)/512

print(np.max(train_locations))
print(np.min(train_locations))

def as_image(image_path):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, (img_height, img_width))
    image = tf.image.convert_image_dtype(image, tf.float32) / 255.0
    return image

def gaussian_label_smoothing(x, num_bins=512, sigma=3):
    # Create a soft probability distribution centered at x using a Gaussian.
    bins = np.arange(num_bins)
    soft_label = scipy.stats.norm.pdf(bins, loc=x, scale=sigma)  # Gaussian centered at x
    soft_label /= soft_label.sum()  # Normalize to sum to 1 (valid probability distribution)
    
    return soft_label

# Function to convert (x, y) coordinate labels into one-hot probability maps
def convert_labels_to_heatmaps(x_coords, y_coords, img_size=512, sigma = 4):
    x_labels = np.array([gaussian_label_smoothing(x, img_size, sigma) for x in x_coords])
    y_labels = np.array([gaussian_label_smoothing(y, img_size, sigma) for y in y_coords])
    return x_labels, y_labels

def create_dataset(paths, locs):
    dataset = tf.data.Dataset.from_tensor_slices((paths, locs))
    dataset = dataset.map(lambda i, xy: (as_image(i), {'x': xy[0], 'y': xy[1]}), num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size)
    dataset = dataset.cache()
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    return dataset

# Convert (x, y) labels into one-hot encoded heatmaps
train_x_heatmaps, train_y_heatmaps = convert_labels_to_heatmaps(train_locations[:, 0] * 512, train_locations[:, 1] * 512)
val_x_heatmaps, val_y_heatmaps = convert_labels_to_heatmaps(val_locations[:, 0] * 512, val_locations[:, 1] * 512)
test_x_heatmaps, test_y_heatmaps = convert_labels_to_heatmaps(test_locations[:, 0] * 512, test_locations[:, 1] * 512)

train_dataset = create_dataset(train_image_paths_locations, (train_x_heatmaps, train_y_heatmaps))
val_dataset = create_dataset(val_image_paths_locations, (val_x_heatmaps, val_y_heatmaps))
test_dataset = create_dataset(test_image_paths_locations, (test_x_heatmaps, test_y_heatmaps))

base_model = tf.keras.applications.MobileNetV3Small(
    input_shape=(img_height, img_width, 3),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = True
for layer in base_model.layers[:-30]:  # Keep most layers frozen
    layer.trainable = False

'''
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=3e-5,
    decay_steps=10000,
    decay_rate=0.9,
    staircase=True
)
'''
'''
lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
    initial_learning_rate=3e-5,
    decay_steps=10000
)
'''

lr_schedule = 1e-3
reduce_lr_on_plateau = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',         # Metric to monitor
    factor=0.5,                 # Factor to reduce learning rate by (new_lr = lr * factor)
    patience=3,                 # Number of epochs with no improvement after which learning rate will be reduced
    verbose=1,                  # Print when the learning rate is reduced
    min_lr=1e-7                 # Lower bound on the learning rate
)

lr = 1e-4
epochs = 100

conv = base_model.output
conv = tf.keras.layers.Conv2D(128, (5, 5), activation='swish', padding='same')(conv)
conv = tf.keras.layers.BatchNormalization()(conv)
conv = tf.keras.layers.Conv2D(64, (3, 3), activation='swish', padding='same')(conv)
conv = tf.keras.layers.BatchNormalization()(conv)
conv = tf.keras.layers.Conv2D(32, (3, 3), activation='swish', padding='same')(conv)
conv = tf.keras.layers.BatchNormalization()(conv)
avg = tf.keras.layers.GlobalAveragePooling2D()(conv)
max_ = tf.keras.layers.GlobalMaxPooling2D()(conv)
spp = tf.keras.layers.Concatenate()([avg, max_])
spp = tf.keras.layers.Dense(128, activation='swish')(spp)  # channel attention


# LOCATION X HEATMAP HEAD
x_head = tf.keras.layers.Dense(256, activation='swish')(spp)  # Output heatmap size (512 bins)
x_head = tf.keras.layers.Dense(512)(x_head)  # Pre-softmax logits
x_head = tf.keras.layers.Softmax(name='x')(x_head)  # Probability distribution

# LOCATION Y HEATMAP HEAD
y_head = tf.keras.layers.Dense(256, activation='swish')(spp)  # Output heatmap size (512 bins)
y_head = tf.keras.layers.Dense(512)(y_head)  # Pre-softmax logits
y_head = tf.keras.layers.Softmax(name='y')(y_head)  # Probability distribution

# Define the model
model = tf.keras.Model(inputs=base_model.input, outputs=(x_head, y_head))

# Define losses (using KL divergence for heatmaps)
losses = { 
    'x': 'kl_divergence', 
    'y': 'kl_divergence' 
}

loss_weights = { 'x': 0.5, 'y': 0.5 }

opt = tf.keras.optimizers.Adam(
    learning_rate=lr_schedule,
)

model.compile(loss=losses, optimizer=opt, loss_weights=loss_weights)

callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5),
    tf.keras.callbacks.ModelCheckpoint(filepath='xy_heatmap_model.keras', monitor='val_loss', save_best_only=True),
    reduce_lr_on_plateau
]

print(train_dataset)
history = model.fit(train_dataset,
                    validation_data = val_dataset,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    callbacks=callbacks)

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper right')
plt.savefig('loss_xy.png')
plt.show()

plt.plot(history.history['x_loss'])
plt.plot(history.history['val_x_loss'])
plt.title('Model X Loss')
plt.ylabel('X KL Divergence')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper right')
plt.savefig('x_loss.png')
plt.show()

plt.plot(history.history['y_loss'])
plt.plot(history.history['val_y_loss'])
plt.title('Model Y Loss')
plt.ylabel('Y KL Divergence')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper right')
plt.savefig('y_loss.png')
plt.show()

model.evaluate(test_dataset)

model = tf.keras.models.load_model("xy_heatmap_model.keras")

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

for image_path in test_image_paths_locations[:20]:  # Test on a few images
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

predictions = model.predict(test_dataset)
model.evaluate(test_dataset)

locations = predictions
x, y = np.array(locations[0]).flatten(), np.array(locations[1]).flatten()
print(np.array(x).shape)
print(np.array(y).shape)
plt.hist2d(x,y, bins=[np.arange(0, 1,12/512),np.arange(0, 1,12/512)])
plt.colorbar()
plt.show()

locations = np.array(test_locations)
x, y = locations[:,0], locations[:, 1]

print(len(x))

plt.hist2d(x,y, bins=[np.arange(0, 1,12/512),np.arange(0, 1,12/512)])
plt.colorbar()
plt.show()

locations = np.array(train_locations)
x, y = locations[:,0], locations[:, 1]

print(len(x))

plt.hist2d(x,y, bins=[np.arange(0, 1,12/512),np.arange(0, 1,12/512)])
plt.colorbar()
plt.show()

best_model = tf.keras.models.load_model('xy_heatmap_model.keras')
predictions = best_model.predict(test_dataset)

locations = predictions
x, y = np.array(locations[0]).flatten(), np.array(locations[1]).flatten()
print(np.array(x).shape)
print(np.array(y).shape)
plt.hist2d(x,y, bins=[np.arange(0, 1,12/512),np.arange(0, 1,12/512)])
plt.colorbar()
plt.show()

locations = np.array(test_locations)
x, y = locations[:,0], locations[:, 1]

print(len(x))

plt.hist2d(x,y, bins=[np.arange(0, 1,12/512),np.arange(0, 1,12/512)])
plt.colorbar()
plt.show()

locations = np.array(train_locations)
x, y = locations[:,0], locations[:, 1]

print(len(x))

plt.hist2d(x,y, bins=[np.arange(0, 1,12/512),np.arange(0, 1,12/512)])
plt.colorbar()
plt.show()

best_model.evaluate(test_dataset)