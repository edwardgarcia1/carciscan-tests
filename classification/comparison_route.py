import pandas as pd
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, classification_report)
import os
import warnings
import ast
from collections import Counter

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# --- Configuration ---
DATASET_PATH = '../data/dataset.csv'
RESULTS_DIR = 'results'
RESULTS_CSV_PATH = os.path.join(RESULTS_DIR, 'route_multilabel_model_comparison.csv')
EXCLUDE_COLUMNS = ['CID', 'Carcinogenicity', 'Categories']
TARGET_COLUMN = 'Route'
MIN_LABEL_OCCURRENCES = 50  # Minimum number of occurrences for a label to be kept

# Add clipping values to handle extreme numbers
max_clip_value = 1e15
min_clip_value = -1e15

# Create directories if they don't exist
os.makedirs(RESULTS_DIR, exist_ok=True)

# --- 1. Load and Prepare Data ---
print("Loading data...")
df = pd.read_csv(DATASET_PATH)
print(f"Initial dataset shape: {df.shape}")

# Remove rows where Route is "['No data']" or NaN
print(f"Removing rows with 'No data' or missing {TARGET_COLUMN}...")
df = df[df[TARGET_COLUMN].notna()]
df = df[df[TARGET_COLUMN] != "['No data']"].copy()
print(f"Dataset shape after removing 'No data' rows: {df.shape}")

# --- 2. Filter Infrequent Labels ---
print("\n--- Filtering Infrequent Route Labels ---")
# Safely convert string representation of list to an actual list of strings
y_list = df[TARGET_COLUMN].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else [])

# Count the occurrences of each label
all_labels = [label for sublist in y_list for label in sublist]
label_counts = Counter(all_labels)
print(f"Found {len(label_counts)} unique route labels in total.")

# Identify labels that meet the minimum occurrence threshold
frequent_labels = {label for label, count in label_counts.items() if count > MIN_LABEL_OCCURRENCES}
labels_to_remove = {label for label, count in label_counts.items() if count <= MIN_LABEL_OCCURRENCES}

print(f"Found {len(labels_to_remove)} labels with {MIN_LABEL_OCCURRENCES} or fewer occurrences.")
if labels_to_remove:
    print("Labels to be removed:", sorted(list(labels_to_remove)))
print(f"Proceeding with {len(frequent_labels)} frequent labels.")

# Filter the route lists in the DataFrame to keep only frequent labels
df['filtered_routes'] = y_list.apply(lambda labels: [label for label in labels if label in frequent_labels])

# Remove rows that now have no labels after filtering
initial_rows = len(df)
df = df[df['filtered_routes'].map(len) > 0].copy()
final_rows = len(df)

print(f"Removed {initial_rows - final_rows} rows that had no frequent labels.")
print(f"Final dataset shape for training: {df.shape}")

# --- 3. Preprocess Features and Target ---
# Define features (X) and target (y) from the cleaned dataframe
X = df.drop(columns=EXCLUDE_COLUMNS + [TARGET_COLUMN, 'filtered_routes'])
y_raw = df['filtered_routes']

# --- Preprocess Target (Multilabel) ---
print("\nPreprocessing target variable for multilabel classification...")
# Use MultiLabelBinarizer to convert the list of labels into a binary matrix
mlb = MultiLabelBinarizer(classes=sorted(list(frequent_labels)))
y = mlb.fit_transform(y_raw)

print(f"Shape of target matrix (y): {y.shape}")
print("Final route labels for model:", mlb.classes_)

# --- Preprocess Features ---
# Handle missing values in features by imputing with the mean
# print(f"\nMissing values in features before imputation: {X.isnull().sum().sum()}")
# X.fillna(X.mean(), inplace=True)
# print(f"Missing values in features after imputation: {X.isnull().sum().sum()}")

# Clip extreme values to prevent errors
print(f"Clipping feature values to the range [{min_clip_value}, {max_clip_value}]...")
X.clip(lower=min_clip_value, upper=max_clip_value, inplace=True)

# --- 4. Split Data ---
# Split data into training and testing sets. No stratification for multilabel y.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"\nTraining set shape: {X_train.shape}")
print(f"Testing set shape: {X_test.shape}")

# --- 5. Define Models (wrapped for Multilabel) ---
models = {
    "XGBoost": MultiOutputClassifier(xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        random_state=42,
        use_label_encoder=False
    )),
    "LightGBM": MultiOutputClassifier(lgb.LGBMClassifier(
        objective='binary',
        metric='binary_logloss',
        random_state=42
    )),
    "CatBoost": MultiOutputClassifier(cb.CatBoostClassifier(
        loss_function='Logloss',
        eval_metric='Logloss',
        random_state=42,
        verbose=0
    ))
}

# --- 6. Train, Evaluate, and Compare Models ---
results = []

for name, model in models.items():
    print(f"\n--- Training and Evaluating {name} ---")

    # Train the model
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate overall metrics using multilabel-appropriate methods
    # 'subset' accuracy requires all labels for a sample to be correct
    accuracy = accuracy_score(y_test, y_pred)
    # 'micro' average aggregates contributions of all labels
    precision = precision_score(y_test, y_pred, average='micro')
    recall = recall_score(y_test, y_pred, average='micro')
    f1 = f1_score(y_test, y_pred, average='micro')

    # Generate detailed classification report
    report = classification_report(y_test, y_pred, target_names=mlb.classes_, output_dict=True, zero_division=0)

    # Store results
    model_results = {
        'Model': name,
        'Accuracy (Exact Match)': accuracy,
        'Precision (Micro)': precision,
        'Recall (Micro)': recall,
        'F1-Score (Micro)': f1
    }

    # Add per-label F1-scores to the results dictionary
    for label_name in mlb.classes_:
        model_results[f'F1-Score_{label_name}'] = report[label_name]['f1-score']

    results.append(model_results)

    # Print metrics to console
    print(f"Accuracy (Exact Match): {accuracy:.4f}")
    print(f"Precision (Micro): {precision:.4f}")
    print(f"Recall (Micro): {recall:.4f}")
    print(f"F1-Score (Micro): {f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=mlb.classes_, zero_division=0))

# --- 7. Save Results to CSV ---
print("\n--- Saving Comparison Results ---")
results_df = pd.DataFrame(results)

# Reorder columns for better readability
column_order = ['Model', 'Accuracy (Exact Match)', 'Precision (Micro)', 'Recall (Micro)', 'F1-Score (Micro)']
per_label_cols = sorted([col for col in results_df.columns if col.startswith('F1-Score_')])
results_df = results_df[column_order + per_label_cols]

print("\nModel Comparison Summary:")
print(results_df.to_string(index=False))

results_df.to_csv(RESULTS_CSV_PATH, index=False)
print(f"\nDetailed comparison results saved to '{RESULTS_CSV_PATH}'")