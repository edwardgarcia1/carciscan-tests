import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, classification_report, confusion_matrix)
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- Configuration ---
DATASET_PATH = '../data/dataset.csv'
MODEL_FILE_PATH = '../models/carcinogenicity.pkl'
RESULTS_DIR = 'results'
PLOT_PATH = os.path.join(RESULTS_DIR, 'carcinogenicity_confusion_matrix.png')
EXCLUDE_COLUMNS = ['CID', 'Categories', 'Route']
TARGET_COLUMN = 'Carcinogenicity'
VALID_CARCINOGENICITY_CLASSES = ['Group 1', 'Group 2A', 'Group 2B', 'Group 3']

# Add clipping values to handle extreme numbers
max_clip_value = 1e15
min_clip_value = -1e15

# Create directories if they don't exist
os.makedirs(os.path.dirname(MODEL_FILE_PATH), exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# --- 1. Load and Prepare Data ---
print("Loading data...")
df = pd.read_csv(DATASET_PATH)
print(f"Initial dataset shape: {df.shape}")

# Remove rows with 'No data' in the target column
df = df[df[TARGET_COLUMN] != 'No data'].copy()
print(f"Dataset shape after removing 'No data' rows: {df.shape}")

# Define features (X) and target (y)
X = df.drop(columns=EXCLUDE_COLUMNS + [TARGET_COLUMN])
y = df[TARGET_COLUMN]

# --- 2. Preprocess Features and Target ---
# Handle missing values in features by imputing with the mean
print(f"Missing values in features before imputation: {X.isnull().sum().sum()}")
X.fillna(X.mean(), inplace=True)
print(f"Missing values in features after imputation: {X.isnull().sum().sum()}")

# Clip extreme values to prevent XGBoost errors
print(f"Clipping feature values to the range [{min_clip_value}, {max_clip_value}]...")
X.clip(lower=min_clip_value, upper=max_clip_value, inplace=True)

# Encode the categorical target variable
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Verify the classes match the expected ones
print("\nTarget variable encoding:")
class_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
print(class_mapping)
assert set(label_encoder.classes_) == set(VALID_CARCINOGENICITY_CLASSES), "Unexpected classes in target variable."

# --- 3. Split Data ---
# Split data into training and testing sets with stratification
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)
print(f"\nTraining set shape: {X_train.shape}")
print(f"Testing set shape: {X_test.shape}")

# --- 4. Train XGBoost Model ---
# Configure XGBoost for multiclass classification
num_classes = len(VALID_CARCINOGENICITY_CLASSES)
model = xgb.XGBClassifier(
    objective='multi:softmax',
    num_class=num_classes,
    eval_metric='mlogloss',
    random_state=42
)

print("\nTraining XGBoost model...")
model.fit(X_train, y_train)
print("Training complete.")

# --- 5. Evaluate Model ---
print("\n--- Model Evaluation ---")
# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

# Print metrics
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision (Weighted): {precision:.4f}")
print(f"Recall (Weighted): {recall:.4f}")
print(f"F1-score (Weighted): {f1:.4f}")

# Print detailed classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# Generate and display confusion matrix
print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Plot and save confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=label_encoder.classes_,
            yticklabels=label_encoder.classes_)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.savefig(PLOT_PATH)
print(f"\nConfusion matrix plot saved to '{PLOT_PATH}'")

# --- 6. Save Model and Encoder ---
# Save the model, encoder, and feature names in a single dictionary
model_data = {
    'model': model,
    'label_encoder': label_encoder,
    'feature_names': X.columns.tolist()
}

with open(MODEL_FILE_PATH, 'wb') as f:
    pickle.dump(model_data, f)

print(f"\nModel, encoder, and feature names saved to '{MODEL_FILE_PATH}'")