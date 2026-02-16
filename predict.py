import numpy as np
import cv2
from tensorflow.keras.models import load_model

# Load trained model
model = load_model("skin_disease_model.h5")

# Class labels (HAM10000 dataset classes)
classes = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']

# Image path (change this to your test image)
image_path = "test.jpg"

# Read and preprocess image
img = cv2.imread(image_path)
img = cv2.resize(img, (64, 64))
img = img / 255.0
img = np.reshape(img, (1, 64, 64, 3))

# Prediction
prediction = model.predict(img)
class_index = np.argmax(prediction)

print("Predicted Disease:", classes[class_index])
print("Confidence:", np.max(prediction))
