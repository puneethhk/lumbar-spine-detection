import pandas as pd
import pydicom
import numpy as np
import os
from skimage.transform import resize

# Constants
IMAGE_SHAPE = (64, 64)  # Reduced size for all images
BATCH_SIZE = 1000  # Number of images to process in a batch

# Load data
train_df = pd.read_csv('D:/spine_classification/data/train.csv')
train_label_coordinates_df = pd.read_csv('D:/spine_classification/data/train_label_coordinates.csv')

# Preprocess images and labels
def preprocess_image(image):
    image_resized = resize(image, IMAGE_SHAPE, anti_aliasing=True)
    return image_resized.flatten()

def save_batch(batch_data, batch_labels, batch_num):
    X = np.array(batch_data)
    y = np.array(batch_labels)
    np.save(f'D:/spine_classification/data/X_batch_{batch_num}.npy', X)
    np.save(f'D:/spine_classification/data/y_batch_{batch_num}.npy', y)

preprocessed_images = []
labels = []
batch_num = 0

for idx, row in train_label_coordinates_df.iterrows():
    file_path = f"D:/spine_classification/data/train_images/{row['study_id']}/{row['series_id']}/{row['instance_number']}.dcm"
    if os.path.exists(file_path):
        try:
            ds = pydicom.dcmread(file_path)
            image = preprocess_image(ds.pixel_array)
            preprocessed_images.append(image)
            labels.append(row['condition_label'])  # Adjust according to your label structure
            if len(preprocessed_images) >= BATCH_SIZE:
                save_batch(preprocessed_images, labels, batch_num)
                preprocessed_images = []
                labels = []
                batch_num += 1
        except Exception as e:
            print(f"Error processing file {file_path}: {e}")

# Save any remaining data
if preprocessed_images:
    save_batch(preprocessed_images, labels, batch_num)
