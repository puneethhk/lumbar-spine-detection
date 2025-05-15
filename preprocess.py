import os
import numpy as np
import pandas as pd
import cv2
import pydicom
# Load and preprocess DICOM images
def preprocess_dicom_image(dicom_path):
    dicom = pydicom.dcmread(dicom_path)
    image = dicom.pixel_array
    image = cv2.resize(image, (224, 224)) # Resize to the model input size
    image = image / np.max(image) # Normalize the image to the range [0, 1]
    return image
# Load dataset
def load_dataset(csv_file, image_dir):
    data = pd.read_csv(csv_file)
    images = []
    labels = []
    for idx, row in data.iterrows():
        dicom_path = os.path.join(image_dir, row['image_name'])
        image = preprocess_dicom_image(dicom_path)
        images.append(image)
        labels.append(row['label']) # Adjust as necessary to match your label format
        images = np.array(images).reshape(-1, 224, 224, 1) # Add channel dimension
        labels = np.array(labels)
        
    return images, labels