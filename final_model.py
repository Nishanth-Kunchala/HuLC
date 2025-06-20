import tensorflow as tf
import pandas as pd
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
# Constants
img_height = 224
img_width = 224
heatmap_size = 512  # Based on your output head
sigma = 1.0         # For soft Gaussian heatmap
image_dir = 'rotated_data/'
csv_path = 'labeling_rotated.csv'

def generate_heatmap(coord, size=512, sigma=1.0):
    """Generate softmaxable 1D heatmap with Gaussian smoothing."""
    x = np.arange(size)
    heatmap = np.exp(-0.5 * ((x - coord) / sigma) ** 2)
    heatmap /= np.sum(heatmap)  # Normalize to make it a probability distribution
    return heatmap.astype(np.float32)

df = pd.read_csv(csv_path)
df.columns = ['x', 'y', 'filepath']

def load_sample(row):
    filepath = os.path.join(image_dir, row['filepath'])
    x = float(row['x'])
    y = float(row['y'])

    x = np.clip(x, 0, heatmap_size - 1)
    y = np.clip(y, 0, heatmap_size - 1)

    return filepath, x, y

samples = [load_sample(row) for _, row in df.iterrows()]

def preprocess_image(image_path):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_png(img, channels=3)
    img = tf.image.resize(img, [img_height, img_width])
    img = tf.cast(img, tf.float32) / 255.0
    return img

def preprocess_sample(image_path, x, y):
    img = preprocess_image(image_path)

    x_heatmap = tf.py_function(func=lambda val: generate_heatmap(val, size=heatmap_size, sigma=sigma), inp=[x], Tout=tf.float32)
    y_heatmap = tf.py_function(func=lambda val: generate_heatmap(val, size=heatmap_size, sigma=sigma), inp=[y], Tout=tf.float32)

    x_heatmap.set_shape([heatmap_size])
    y_heatmap.set_shape([heatmap_size])

    return img, {"x": x_heatmap, "y": y_heatmap}

filepaths = [s[0] for s in samples]
x_coords = [s[1] for s in samples]
y_coords = [s[2] for s in samples]

dataset = tf.data.Dataset.from_tensor_slices((filepaths, x_coords, y_coords))
dataset = dataset.map(preprocess_sample, num_parallel_calls=tf.data.AUTOTUNE)

batch_size = 32
# Full list of samples
total = len(samples)
train_frac = 0.8
val_frac = 0.1
test_frac = 0.1

train_size = int(total * train_frac)
val_size = int(total * val_frac)
test_size = total - train_size - val_size  # remaining

# Shuffle once for deterministic split
dataset = dataset.shuffle(total, reshuffle_each_iteration=False)

# Split
train_dataset = dataset.take(train_size)
val_dataset = dataset.skip(train_size).take(val_size)
test_dataset = dataset.skip(train_size + val_size)

# Batch and prefetch
train_dataset = train_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
val_dataset = val_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
test_dataset = test_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)


base_model = tf.keras.applications.MobileNetV3Small(
    input_shape=(img_height, img_width, 3),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False

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
conv = tf.keras.layers.Conv2D(64, (5, 5), activation='swish', padding='same')(conv)
conv = tf.keras.layers.Conv2D(64, (3, 3), activation='swish', padding='same')(conv)
conv = tf.keras.layers.Conv2D(32, (3, 3), activation='swish', padding='same')(conv)
spp1 = tf.keras.layers.GlobalAveragePooling2D()(conv)
spp2 = tf.keras.layers.GlobalMaxPooling2D()(conv)
spp = tf.keras.layers.Concatenate()([spp1, spp2])  # Combine both features
flatten = tf.keras.layers.Flatten()(spp)

# LOCATION X HEATMAP HEAD
regression = tf.keras.layers.Dense(16, activation='swish')(flatten)
x_head = tf.keras.layers.Dense(8, activation='swish')(regression)
x_head = tf.keras.layers.Dense(512, activation='swish')(x_head)  # Output heatmap size (512 bins)
x_head = tf.keras.layers.Dense(512, activation='linear')(x_head)  # Pre-softmax logits
x_head = tf.keras.layers.Softmax(name='x')(x_head)  # Probability distribution

# LOCATION Y HEATMAP HEAD
regression = tf.keras.layers.Dense(8, activation='swish')(flatten)
y_head = tf.keras.layers.Dense(4, activation='swish')(regression)
y_head = tf.keras.layers.Dense(512, activation='swish')(y_head)  # Output heatmap size (512 bins)
y_head = tf.keras.layers.Dense(512, activation='linear')(y_head)  # Pre-softmax logits
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

