import pandas as pd
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, classification_report)
import os
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# --- Configuration ---
DATASET_PATH = '../data/dataset.csv'
RESULTS_DIR = 'results'
OVERALL_RESULTS_PATH = os.path.join(RESULTS_DIR, 'carcinogenicity_model_comparison_overall.csv')
PER_CLASS_RESULTS_DIR = os.path.join(RESULTS_DIR, 'per_class_metrics')
EXCLUDE_COLUMNS = ['CID', 'Categories', 'Route']
TARGET_COLUMN = 'Carcinogenicity'
VALID_CARCINOGENICITY_CLASSES = ['Group 1', 'Group 2A', 'Group 2B', 'Group 3']

# Add clipping values to handle extreme numbers
max_clip_value = 1e15
min_clip_value = -1e15

# Create directories if they don't exist
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PER_CLASS_RESULTS_DIR, exist_ok=True)

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
# Clip extreme values to prevent errors
print(f"Clipping feature values to the range [{min_clip_value}, {max_clip_value}]...")
X.clip(lower=min_clip_value, upper=max_clip_value, inplace=True)

# Encode the categorical target variable
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# --- 3. Split Data ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)
print(f"\nTraining set shape: {X_train.shape}")
print(f"Testing set shape: {X_test.shape}")

# --- 4. Define Models ---
models = {
    "XGBoost": xgb.XGBClassifier(
        objective='multi:softmax',
        num_class=len(VALID_CARCINOGENICITY_CLASSES),
        eval_metric='mlogloss',
        random_state=42
    ),
    "LightGBM": lgb.LGBMClassifier(
        objective='multiclass',
        metric='multi_logloss',
        num_class=len(VALID_CARCINOGENICITY_CLASSES),
        random_state=42
    ),
    "CatBoost": cb.CatBoostClassifier(
        loss_function='MultiClass',
        eval_metric='MultiClass',
        random_state=42,
        verbose=False
    )
}

# --- 5. Initialize Data Structures for Results ---
overall_results = []
per_class_results = {class_name: [] for class_name in label_encoder.classes_}

# --- 6. Train, Evaluate, and Compare Models ---
for name, model in models.items():
    print(f"\n--- Training and Evaluating {name} ---")

    # Train the model
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate overall metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    # Generate detailed classification report
    report = classification_report(y_test, y_pred, target_names=label_encoder.classes_, output_dict=True)

    # Store overall results (ROUNDED TO 4 DECIMAL PLACES)
    overall_results.append({
        'Model': name,
        'Accuracy': round(accuracy, 4),
        'Precision (Weighted)': round(precision, 4),
        'Recall (Weighted)': round(recall, 4),
        'F1-Score (Weighted)': round(f1, 4)
    })

    # Store per-class results (ROUNDED TO 4 DECIMAL PLACES)
    for class_name in label_encoder.classes_:
        per_class_results[class_name].append({
            'Model': name,
            'Precision': round(report[class_name]['precision'], 4),
            'Recall': round(report[class_name]['recall'], 4),
            'F1-Score': round(report[class_name]['f1-score'], 4),
            'Support': report[class_name]['support']  # Support is an integer, no rounding needed
        })

    # Print metrics to console (formatted to 4 decimal places)
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision (Weighted): {precision:.4f}")
    print(f"Recall (Weighted): {recall:.4f}")
    print(f"F1-Score (Weighted): {f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_, digits=4))

# --- 7. Save Results to CSV Files ---
print("\n--- Saving Results ---")

# Save overall results
overall_df = pd.DataFrame(overall_results)
overall_df.to_csv(OVERALL_RESULTS_PATH, index=False)
print(f"Overall results saved to '{OVERALL_RESULTS_PATH}'")

# Display overall results
print("\nModel Comparison Summary (Overall):")
print(overall_df.to_string(index=False))

# Save per-class results
for class_name, results_list in per_class_results.items():
    # Create DataFrame for this class
    class_df = pd.DataFrame(results_list)

    # Create safe filename (replace spaces and special characters)
    safe_class_name = class_name.replace(' ', '_').replace('/', '_')
    class_csv_path = os.path.join(PER_CLASS_RESULTS_DIR, f'carcinogenicity_{safe_class_name}_metrics.csv')

    # Save to CSV
    class_df.to_csv(class_csv_path, index=False)
    print(f"\nPer-class results for '{class_name}' saved to '{class_csv_path}'")

    # Display per-class results
    print(f"\nMetrics for {class_name}:")
    print(class_df.to_string(index=False))

# --- 8. Create Summary of All Per-Class Results ---
print("\n--- Creating Summary of All Per-Class Results ---")
summary_dfs = []

for class_name in label_encoder.classes_:
    class_df = pd.DataFrame(per_class_results[class_name])
    # Add class name as a column for identification
    class_df['Class'] = class_name
    summary_dfs.append(class_df)

# Combine all per-class results into one summary DataFrame
all_classes_summary = pd.concat(summary_dfs, ignore_index=True)

# Reorder columns
summary_column_order = ['Class', 'Model', 'Precision', 'Recall', 'F1-Score', 'Support']
all_classes_summary = all_classes_summary[summary_column_order]

# Save summary
summary_path = os.path.join(RESULTS_DIR, 'carcinogenicity_all_classes_summary.csv')
all_classes_summary.to_csv(summary_path, index=False)
print(f"\nAll classes summary saved to '{summary_path}'")

# Display summary
print("\nAll Classes Summary:")
print(all_classes_summary.to_string(index=False))

print("\n--- Analysis Complete ---")
print(f"Total files created: {1 + len(label_encoder.classes_) + 1}")  # overall + per-class + summary