import numpy as np
import pydicom
from skimage.transform import resize
from skimage import io
import os

# Constants
IMAGE_SHAPE = (64, 64)  # Ensure this matches the shape used in training
DISPLAY_SHAPE = (256, 256)  # Shape for displaying the image

def preprocess_image(dicom_file_path):
    ds = pydicom.dcmread(dicom_file_path)
    image = ds.pixel_array
    image_resized = resize(image, IMAGE_SHAPE, anti_aliasing=True)
    return image_resized.flatten()

def dicom_to_image(dicom_file_path, output_path):
    ds = pydicom.dcmread(dicom_file_path)
    image = ds.pixel_array
    image_resized = resize(image, DISPLAY_SHAPE, anti_aliasing=True)
    
    # Normalize the image to 0-255 and convert to uint8
    image_resized = (255 * (image_resized - np.min(image_resized)) / np.ptp(image_resized)).astype(np.uint8)
    
    io.imsave(output_path, image_resized)

def predict(model, image):
    prediction = model.predict([image])
    return prediction