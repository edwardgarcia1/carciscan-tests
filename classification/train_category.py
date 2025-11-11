import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, classification_report, multilabel_confusion_matrix)
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import os
import ast
from collections import Counter

# --- Configuration ---
DATASET_PATH = '../data/dataset.csv'
MODEL_FILE_PATH = '../models/category.pkl'
RESULTS_DIR = 'results'
PLOT_PATH = os.path.join(RESULTS_DIR, 'category_confusion_matrices.png')
EXCLUDE_COLUMNS = ['Carcinogenicity', 'CID', 'Route']
TARGET_COLUMN = 'Categories'
MIN_LABEL_OCCURRENCES = 50 # Minimum number of occurrences for a label to be kept

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

# Remove rows where Categories is "['No data']" or NaN
print("Removing rows with 'No data' or missing Categories...")
df = df[df[TARGET_COLUMN].notna()]
df = df[df[TARGET_COLUMN] != "['No data']"].copy()
print(f"Dataset shape after removing 'No data' rows: {df.shape}")

# --- 2. Filter Infrequent Labels ---
print("\n--- Filtering Infrequent Labels ---")
# Safely convert string representation of list to an actual list of strings
y_list = df[TARGET_COLUMN].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else [])

# Count the occurrences of each label
all_labels = [label for sublist in y_list for label in sublist]
label_counts = Counter(all_labels)
print(f"Found {len(label_counts)} unique labels in total.")

# Identify labels that meet the minimum occurrence threshold
frequent_labels = {label for label, count in label_counts.items() if count > MIN_LABEL_OCCURRENCES}
labels_to_remove = {label for label, count in label_counts.items() if count <= MIN_LABEL_OCCURRENCES}

print(f"Found {len(labels_to_remove)} labels with {MIN_LABEL_OCCURRENCES} or fewer occurrences.")
if labels_to_remove:
    print("Labels to be removed:", sorted(list(labels_to_remove)))
print(f"Proceeding with {len(frequent_labels)} frequent labels.")

# Filter the category lists in the DataFrame to keep only frequent labels
df['filtered_categories'] = y_list.apply(lambda labels: [label for label in labels if label in frequent_labels])

# Remove rows that now have no labels after filtering
initial_rows = len(df)
df = df[df['filtered_categories'].map(len) > 0].copy()
final_rows = len(df)

print(f"Removed {initial_rows - final_rows} rows that had no frequent labels.")
print(f"Final dataset shape for training: {df.shape}")

# --- 3. Preprocess Features and Target ---

# Define features (X) and target (y) from the cleaned dataframe
X = df.drop(columns=EXCLUDE_COLUMNS + [TARGET_COLUMN, 'filtered_categories'])
y_raw = df['filtered_categories']

# --- Preprocess Target (Multilabel) ---
print("\nPreprocessing target variable for multilabel classification...")
# Use MultiLabelBinarizer to convert the list of labels into a binary matrix
mlb = MultiLabelBinarizer(classes=sorted(list(frequent_labels))) # Ensure consistent class order
y = mlb.fit_transform(y_raw)

print(f"Shape of target matrix (y): {y.shape}")
print("Final labels for model:", mlb.classes_)

# --- Preprocess Features ---
# Handle missing values in features by imputing with the mean
# print(f"\nMissing values in features before imputation: {X.isnull().sum().sum()}")
# X.fillna(X.mean(), inplace=True)
# print(f"Missing values in features after imputation: {X.isnull().sum().sum()}")

# Clip extreme values to prevent XGBoost errors
print(f"Clipping feature values to the range [{min_clip_value}, {max_clip_value}]...")
X.clip(lower=min_clip_value, upper=max_clip_value, inplace=True)

# --- 4. Split Data ---
# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"\nTraining set shape: {X_train.shape}")
print(f"Testing set shape: {X_test.shape}")

# --- 5. Train Multilabel Model ---
# We use MultiOutputClassifier to wrap a binary classifier for each label
base_classifier = xgb.XGBClassifier(
    objective='binary:logistic',
    eval_metric='logloss',
    random_state=42
)

model = MultiOutputClassifier(base_classifier)

print("\nTraining Multilabel XGBoost model...")
model.fit(X_train, y_train)
print("Training complete.")

# --- 6. Evaluate Model ---
print("\n--- Model Evaluation ---")
# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate metrics using multilabel-appropriate averages
accuracy = accuracy_score(y_test, y_pred)
precision_micro = precision_score(y_test, y_pred, average='micro')
recall_micro = recall_score(y_test, y_pred, average='micro')
f1_micro = f1_score(y_test, y_pred, average='micro')
f1_macro = f1_score(y_test, y_pred, average='macro')

# Print metrics
print(f"Subset Accuracy (Exact Match): {accuracy:.4f}")
print(f"Micro-Averaged Precision: {precision_micro:.4f}")
print(f"Micro-Averaged Recall: {recall_micro:.4f}")
print(f"Micro-Averaged F1-score: {f1_micro:.4f}")
print(f"Macro-Averaged F1-score: {f1_macro:.4f}")

# Print detailed classification report for each label
print("\nClassification Report (per label):")
print(classification_report(y_test, y_pred, target_names=mlb.classes_, zero_division=0))

# Generate and display multilabel confusion matrices
print("\nMultilabel Confusion Matrices (per label):")
mcm = multilabel_confusion_matrix(y_test, y_pred)

# Plot and save confusion matrices for each label
num_labels = len(mlb.classes_)
ncols = 4
nrows = int(np.ceil(num_labels / ncols))

fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(20, 5 * nrows))
axes = axes.ravel()

for i, (label, matrix) in enumerate(zip(mlb.classes_, mcm)):
    sns.heatmap(matrix, annot=True, fmt='d', cmap='Blues', ax=axes[i], cbar=False)
    axes[i].set_title(f'{label}')
    axes[i].set_xlabel('Predicted')
    axes[i].set_ylabel('Actual')
    axes[i].set_xticklabels(['0', '1'])
    axes[i].set_yticklabels(['0', '1'])

# Hide any unused subplots
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.savefig(PLOT_PATH)
print(f"\nConfusion matrices plot saved to '{PLOT_PATH}'")

# --- 7. Save Model and Encoder ---
# Save the model, binarizer, and feature names in a single dictionary
model_data = {
    'model': model,
    'multi_label_binarizer': mlb,
    'feature_names': X.columns.tolist()
}

with open(MODEL_FILE_PATH, 'wb') as f:
    pickle.dump(model_data, f)

print(f"\nModel, binarizer, and feature names saved to '{MODEL_FILE_PATH}'")