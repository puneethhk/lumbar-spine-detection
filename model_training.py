import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# Constants
BATCH_SIZE = 1000
data_dir = 'D:/spine_classification/data'

# Load preprocessed data in batches
def load_data_in_batches(data_dir, batch_size):
    X_batches = []
    y_batches = []
    batch_files = [f for f in os.listdir(data_dir) if f.startswith('X_batch_')]
    for batch_file in batch_files:
        X_batch = np.load(os.path.join(data_dir, batch_file))
        y_batch = np.load(os.path.join(data_dir, batch_file.replace('X_batch_', 'y_batch_')))
        X_batches.append(X_batch)
        y_batches.append(y_batch)
    return np.concatenate(X_batches), np.concatenate(y_batches)

X, y = load_data_in_batches(data_dir, BATCH_SIZE)



# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

# Save model
joblib.dump(model, 'D:/spine_classification/backend/model.pkl')
