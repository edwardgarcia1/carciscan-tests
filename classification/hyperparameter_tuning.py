import pandas as pd
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, classification_report)
import os
import warnings
import numpy as np

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# --- Configuration ---
DATASET_PATH = '../data/dataset.csv'
RESULTS_DIR = 'results'
OVERALL_RESULTS_PATH = os.path.join(RESULTS_DIR, 'carcinogenicity_model_comparison_grouped.csv')
PER_CLASS_RESULTS_DIR = os.path.join(RESULTS_DIR, 'per_class_metrics')
EXCLUDE_COLUMNS = ['CID', 'Categories', 'Route']
TARGET_COLUMN = 'Carcinogenicity'
VALID_CARCINOGENICITY_CLASSES = ['Group 1', 'Group 2A', 'Group 2B', 'Group 3']

# --- New Configuration for Tuning and Output ---
TOP_N_CONFIGS = 4  # Number of best tuned configurations to report
MODEL_ORDER = ['XGBoost', 'LightGBM', 'CatBoost']
N_ITER_SEARCH = 50  # Number of parameter settings sampled in RandomizedSearchCV

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

# --- 4. Define Models and Hyperparameter Search Spaces ---
MODEL_CLASSES = {
    "XGBoost": xgb.XGBClassifier,
    "LightGBM": lgb.LGBMClassifier,
    "CatBoost": cb.CatBoostClassifier
}

HYPERPARAMETER_GRIDS = {
    "XGBoost": {
        'n_estimators': [100, 200, 500, 1000],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'max_depth': [3, 5, 7, 9],
        # 'subsample': [0.6, 0.8, 1.0],
        # 'colsample_bytree': [0.6, 0.8, 1.0]
    },
    "LightGBM": {
        'n_estimators': [100, 200, 500, 1000],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'max_depth': [3, 5, 7, 9],  # -1 means no limit
        # 'subsample': [0.6, 0.8, 1.0],
        # 'colsample_bytree': [0.6, 0.8, 1.0]
    },
    "CatBoost": {
        'iterations': [100, 200, 500, 1000],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'depth': [4, 6, 8, 10],
        # 'subsample': [0.6, 0.8, 1.0],
        # 'colsample_bylevel': [0.6, 0.8, 1.0]
    }
}

# --- 5. Initialize Data Structures for Results ---
all_results = []
per_class_results = {}  # Will be populated later

# --- 6. Train, Tune, Evaluate, and Compare Models ---
for model_name in MODEL_ORDER:
    print(f"\n--- Processing Model: {model_name} ---")
    model_class = MODEL_CLASSES[model_name]

    # --- Part A: Evaluate Default Parameters ---
    print(f"Evaluating {model_name} with DEFAULT parameters...")
    default_model = model_class(
        objective='multi:softmax' if model_name == "XGBoost" else (
            'multiclass' if model_name == "LightGBM" else 'MultiClass'),
        num_class=len(VALID_CARCINOGENICITY_CLASSES) if model_name != "CatBoost" else None,
        eval_metric='mlogloss' if model_name == "XGBoost" else (
            'multi_logloss' if model_name == "LightGBM" else 'MultiClass'),
        random_state=42,
        verbose=False if model_name == "CatBoost" else 0
    )
    default_model.fit(X_train, y_train)
    y_pred = default_model.predict(X_test)

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    report = classification_report(y_test, y_pred, target_names=label_encoder.classes_, output_dict=True)

    # Store default results
    result_entry = {
        'Model': model_name,
        'Configuration_Type': 'Default',
        'Parameters': '{}',
        'Rank': 0,
        'Accuracy': round(accuracy, 4),
        'Precision (Weighted)': round(precision, 4),
        'Recall (Weighted)': round(recall, 4),
        'F1-Score (Weighted)': round(f1, 4)
    }
    all_results.append(result_entry)
    per_class_results[f"{model_name}_Default"] = report

    print(f"Default Accuracy: {accuracy:.4f}")

    # --- Part B: Hyperparameter Tuning and Evaluation ---
    print(f"Running Hyperparameter Search for {model_name}...")
    base_model_for_search = model_class(
        objective='multi:softmax' if model_name == "XGBoost" else (
            'multiclass' if model_name == "LightGBM" else 'MultiClass'),
        num_class=len(VALID_CARCINOGENICITY_CLASSES) if model_name != "CatBoost" else None,
        eval_metric='mlogloss' if model_name == "XGBoost" else (
            'multi_logloss' if model_name == "LightGBM" else 'MultiClass'),
        random_state=42,
        verbose=False if model_name == "CatBoost" else 0
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    search = RandomizedSearchCV(
        estimator=base_model_for_search,
        param_distributions=HYPERPARAMETER_GRIDS[model_name],
        n_iter=N_ITER_SEARCH,
        scoring='f1_weighted',
        cv=cv,
        random_state=42,
        n_jobs=-1,
        verbose=0
    )
    search.fit(X_train, y_train)

    # Get all results from the search and sort them
    cv_results_df = pd.DataFrame(search.cv_results_)
    cv_results_df.sort_values(by='rank_test_score', ascending=True, inplace=True)

    print(f"Evaluating top {TOP_N_CONFIGS} tuned configurations...")
    for i in range(TOP_N_CONFIGS):
        rank = i + 1
        best_params = cv_results_df.iloc[i]['params']

        # Train a new model with the best parameters on the full training set
        tuned_model = model_class(**best_params, random_state=42, verbose=False if model_name == "CatBoost" else 0)
        tuned_model.fit(X_train, y_train)
        y_pred_tuned = tuned_model.predict(X_test)

        # Calculate metrics for the tuned model
        accuracy_tuned = accuracy_score(y_test, y_pred_tuned)
        precision_tuned = precision_score(y_test, y_pred_tuned, average='weighted')
        recall_tuned = recall_score(y_test, y_pred_tuned, average='weighted')
        f1_tuned = f1_score(y_test, y_pred_tuned, average='weighted')
        report_tuned = classification_report(y_test, y_pred_tuned, target_names=label_encoder.classes_,
                                             output_dict=True)

        # Store tuned results
        tuned_result_entry = {
            'Model': model_name,
            'Configuration_Type': 'Tuned',
            'Parameters': str(best_params),
            'Rank': rank,
            'Accuracy': round(accuracy_tuned, 4),
            'Precision (Weighted)': round(precision_tuned, 4),
            'Recall (Weighted)': round(recall_tuned, 4),
            'F1-Score (Weighted)': round(f1_tuned, 4)
        }
        all_results.append(tuned_result_entry)
        per_class_results[f"{model_name}_Tuned_Rank_{rank}"] = report_tuned

        print(f"Tuned Rank {rank} Accuracy: {accuracy_tuned:.4f}")

# --- 7. Save Grouped Results to CSV File ---
print("\n--- Saving Grouped Results ---")

# Create DataFrame from all collected results
final_df = pd.DataFrame(all_results)

# Enforce the model order for correct sorting
final_df['Model'] = pd.Categorical(final_df['Model'], categories=MODEL_ORDER, ordered=True)

# Sort by Rank first, then by Model
sorted_df = final_df.sort_values(by=['Rank', 'Model'])

# Define the final column order and drop the helper 'Rank' column
output_columns = [
    'Configuration_Type',
    'Parameters',
    'Model',
    'Accuracy',
    'Precision (Weighted)',
    'Recall (Weighted)',
    'F1-Score (Weighted)'
]
final_output_df = sorted_df[output_columns]

# Save the grouped and sorted results
final_output_df.to_csv(OVERALL_RESULTS_PATH, index=False)
print(f"Grouped results saved to '{OVERALL_RESULTS_PATH}'")

# Display the grouped results
print("\nModel Comparison Summary (Grouped by Configuration Rank):")
print(final_output_df.to_string(index=False))

# --- 8. Save Per-Class Results (for each configuration) ---
print("\n--- Saving Per-Class Results ---")
for config_name, report in per_class_results.items():
    class_df = pd.DataFrame(report).transpose()

    # Create safe filename
    safe_config_name = config_name.replace(' ', '_').replace('/', '_')
    class_csv_path = os.path.join(PER_CLASS_RESULTS_DIR, f'{safe_config_name}_metrics.csv')

    # Save to CSV
    class_df.to_csv(class_csv_path)
    # print(f"Per-class results for '{config_name}' saved to '{class_csv_path}'") # Uncomment for verbose output

print("\n--- Analysis Complete ---")