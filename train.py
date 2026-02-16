import os
import numpy as np
import pandas as pd
import cv2

from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# =========================
# Dataset Paths
# =========================

image_folder1 = "dataset/images/HAM10000_images_part_1"
image_folder2 = "dataset/images/HAM10000_images_part_2"
csv_path = "dataset/HAM10000_metadata.csv"

# =========================
# Load CSV
# =========================

data = pd.read_csv(csv_path)

print("Dataset Loaded")
print(data.head())

# =========================
# Image Processing
# =========================

IMG_SIZE = 64

images = []
labels = []

# Disease labels mapping
label_dict = {label: idx for idx, label in enumerate(data['dx'].unique())}

for index, row in data.iterrows():

    img_name = row['image_id'] + ".jpg"
    label = row['dx']

    img_path1 = os.path.join(image_folder1, img_name)
    img_path2 = os.path.join(image_folder2, img_name)

    # Check which folder contains image
    if os.path.exists(img_path1):
        img_path = img_path1
    elif os.path.exists(img_path2):
        img_path = img_path2
    else:
        continue

    img = cv2.imread(img_path)

    if img is None:
        continue

    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

    images.append(img)
    labels.append(label_dict[label])

print("Total Images Loaded:", len(images))

# Convert to numpy
X = np.array(images) / 255.0
y = to_categorical(labels)

# =========================
# Train Test Split
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training Samples:", X_train.shape)
print("Testing Samples:", X_test.shape)

# =========================
# CNN Model
# =========================

model = Sequential()

model.add(Conv2D(32, (3,3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)))
model.add(MaxPooling2D(2,2))

model.add(Conv2D(64, (3,3), activation='relu'))
model.add(MaxPooling2D(2,2))

model.add(Flatten())

model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(len(label_dict), activation='softmax'))

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# =========================
# Training
# =========================

history = model.fit(
    X_train,
    y_train,
    epochs=10,
    validation_data=(X_test, y_test),
    batch_size=32
)

# =========================
# Evaluation
# =========================

loss, acc = model.evaluate(X_test, y_test)

print("Test Accuracy:", acc)

# =========================
# Save Model
# =========================

model.save("skin_disease_model.h5")

print("Model Saved Successfully ✅")
