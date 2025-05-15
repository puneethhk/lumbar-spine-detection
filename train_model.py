import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from preprocess import load_dataset
def build_model():
     
     model = Sequential([
          Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 1)),
          MaxPooling2D((2, 2)),
          Conv2D(64, (3, 3), activation='relu'),
          MaxPooling2D((2, 2)),
          Conv2D(128, (3, 3), activation='relu'),
          MaxPooling2D((2, 2)),
          Flatten(),
          Dense(128, activation='relu'),
          Dropout(0.5),
          Dense(15, activation='sigmoid') 
])
     model.compile(optimizer=Adam(learning_rate=0.001),
             loss='binary_crossentropy',
             metrics=['accuracy']) 
     return model
images, labels = load_dataset('dataset/dataset.csv', 'dataset/dicom_files')
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)
model = build_model()
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))
model.save('model/LUMBAR PROJECT.h5')