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

path = r'C:\Users\Nishanth Kunchala\Documents\Programming\Research\HuLC'

df = pd.read_csv(os.path.join(path, 'labeling_rotated.csv'))
print(df.head())

n_imgs = len(df)
n_imgs

plt.hist(df['x'], bins=30)
plt.xlabel('X Value')
plt.ylabel('Count')
plt.title('X Value Histogram')
plt.show()

plt.hist(df['y'], bins=30)
plt.xlabel('Y Value')
plt.ylabel('Count')
plt.title('Y Value Histogram')
plt.show()

x, y = df['x'], df['y']
print(x)
print(y)

plt.hist2d(x,y, bins=[np.arange(0, 512, 12),np.arange(0, 512, 12)])
plt.colorbar()
plt.show()