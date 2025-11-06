import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, classification_report,
                             confusion_matrix, make_scorer)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os

warnings.filterwarnings('ignore')


def model_agnostic_feature_selection(X, y, feature_names, n_features):
    """
    Comprehensive model-agnostic feature selection for molecular features
    """
    import pandas as pd
    import numpy as np
    from sklearn.feature_selection import VarianceThreshold, SelectKBest, mutual_info_classif

    # Convert to DataFrame for easier handling
    df = pd.DataFrame(X, columns=feature_names)

    # Step 1: Remove low variance features
    selector_var = VarianceThreshold(threshold=0.01)
    df_var = pd.DataFrame(
        selector_var.fit_transform(df),
        columns=df.columns[selector_var.get_support()]
    )

    if df_var.empty:
        print("Warning: All features removed by variance threshold. Using original features.")
        df_var = df.copy()

    # Step 2: Remove highly correlated features
    if df_var.shape[1] > 1:  # Only if more than 1 feature remains
        corr_matrix = df_var.corr().abs()
        upper_triangle = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        to_drop = [col for col in upper_triangle.columns
                   if any(upper_triangle[col] > 0.95)]
        df_uncorr = df_var.drop(columns=to_drop)
    else:
        df_uncorr = df_var

    if df_uncorr.empty:
        print("Warning: All features removed by correlation filtering. Using variance-filtered features.")
        df_uncorr = df_var

    # Step 3: Select top features using mutual information
    available_features = df_uncorr.shape[1]
    n_select = min(n_features, available_features)

    if n_select <= 0:
        print("Warning: No features available. Using original features.")
        return X, feature_names, None

    selector_mi = SelectKBest(
        score_func=mutual_info_classif,
        k=n_select
    )
    X_final = selector_mi.fit_transform(df_uncorr.values, y)
    final_features = df_uncorr.columns[selector_mi.get_support()].tolist()

    return X_final, final_features, selector_mi


class MLClassifierComparator:
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.models = {
            'Random Forest': RandomForestClassifier(random_state=random_state),
            'XGBoost': XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='mlogloss'),
            'SVM': SVC(random_state=random_state, probability=True)
        }
        self.results = {}
        self.feature_results = {}
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.original_X_train = None
        self.original_X_test = None
        self.y_train = None
        self.y_test = None
        self.feature_names = None

    def prepare_data(self, df, target_col, feature_cols=None, test_size=0.2):
        """
        Prepare data for classification, including handling infinities and extreme values.
        """
        # Select features
        if feature_cols is None:
            feature_cols = [col for col in df.columns if col not in [target_col, 'CID']]

        self.feature_names = feature_cols
        X = df[feature_cols].copy()
        y = df[target_col].copy()

        max_clip_value = 1e15
        min_clip_value = -1e15

        # Check if any values exceed the clip points
        if (X > max_clip_value).any().any() or (X < min_clip_value).any().any():
            print(f"Clipping extreme values to the range [{min_clip_value}, {max_clip_value}]")
            X = X.clip(lower=min_clip_value, upper=max_clip_value)

        X.replace([np.inf, -np.inf], np.nan, inplace=True)

        # Now, handle all NaN values (including the former infinities) by filling with the median
        X = X.fillna(X.median())

        # --- STEP 3: Encode Target Variable ---
        if y.dtype == 'object':
            y = self.label_encoder.fit_transform(y)

        # --- STEP 4: Split Data ---
        self.original_X_train, self.original_X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )

        print(f"Original training set shape: {self.original_X_train.shape}")
        print(f"Original test set shape: {self.original_X_test.shape}")
        print(f"Classes: {np.unique(y)}")
        print(f"Class distribution: {np.bincount(y)}")

    def evaluate_model(self, model, X_train, X_test, y_train, y_test, model_name):
        """
        Evaluate a single model with multiple metrics
        """
        # Fit model
        model.fit(X_train, y_train)

        # Predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None

        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='macro', zero_division=0),
            'recall': recall_score(y_test, y_pred, average='macro', zero_division=0),
            'f1': f1_score(y_test, y_pred, average='macro', zero_division=0),
        }

        # Add AUC if possible (for binary classification or multi-class with proba)
        if y_pred_proba is not None:
            if len(np.unique(y_test)) == 2:
                metrics['auc'] = roc_auc_score(y_test, y_pred_proba[:, 1])
            else:
                try:
                    metrics['auc'] = roc_auc_score(y_test, y_pred_proba, multi_class='ovr')
                except:
                    metrics['auc'] = np.nan

        # Cross-validation scores
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='f1_weighted')
        metrics['cv_mean'] = cv_scores.mean()
        metrics['cv_std'] = cv_scores.std()

        return metrics, y_pred

    def compare_models_with_feature_selection(self, feature_counts=[10, 25, 50, 100, 125, 150, 200, 'all']):
        """
        Compare models across different numbers of selected features
        """
        print("Starting feature selection comparison...")

        # Handle 'all' case
        if 'all' in feature_counts:
            feature_counts.remove('all')
            feature_counts.append(self.original_X_train.shape[1])
            feature_counts = sorted(list(set(feature_counts)))  # Remove duplicates and sort

        for n_features in feature_counts:
            print(f"\\n=== Evaluating with {n_features} features ===")

            # Apply feature selection
            if n_features >= self.original_X_train.shape[1]:
                print(f"Using all {self.original_X_train.shape[1]} features (no selection needed)")
                X_train_selected = self.original_X_train.copy()
                X_test_selected = self.original_X_test.copy()
                selected_features = self.feature_names
            else:
                X_train_selected, selected_features, selector = model_agnostic_feature_selection(
                    self.original_X_train, self.y_train, self.feature_names, n_features
                )
                # Apply same selection to test set
                if n_features >= len(selected_features):
                    # Use the actual selected features
                    X_test_selected = self.original_X_test[selected_features].values
                else:
                    # This shouldn't happen, but just in case
                    X_test_selected = self.original_X_test.iloc[:, :n_features].values
                    selected_features = self.feature_names[:n_features]

            print(f"Selected {len(selected_features)} features")

            # Scale the selected features
            scaler_selected = StandardScaler()
            X_train_scaled = scaler_selected.fit_transform(X_train_selected)
            X_test_scaled = scaler_selected.transform(X_test_selected)

            # Store results for this feature count
            self.feature_results[n_features] = {
                'features': selected_features,
                'X_train': X_train_selected,
                'X_test': X_test_selected,
                'X_train_scaled': X_train_scaled,
                'X_test_scaled': X_test_scaled,
                'model_results': {}
            }

            # Evaluate all models with selected features
            for name, model in self.models.items():
                print(f"  Evaluating {name}...")

                # Use scaled data for models that benefit from it
                if name not in ['Random Forest', 'XGBoost']:
                    X_train, X_test = X_train_scaled, X_test_scaled
                else:
                    X_train, X_test = X_train_selected, X_test_selected

                try:
                    metrics, y_pred = self.evaluate_model(
                        model, X_train, X_test, self.y_train, self.y_test, name
                    )
                    self.feature_results[n_features]['model_results'][name] = metrics
                    print(f"    {name}: F1={metrics['f1']:.4f}, Accuracy={metrics['accuracy']:.4f}")
                except Exception as e:
                    print(f"    {name}: Failed with error - {str(e)}")
                    self.feature_results[n_features]['model_results'][name] = {
                        'accuracy': 0, 'precision': 0, 'recall': 0, 'f1': 0, 'auc': 0, 'cv_mean': 0, 'cv_std': 0
                    }

    def get_best_feature_count(self, metric='f1'):
        """
        Find the best number of features based on average performance across models
        """
        avg_scores = {}
        for n_features, results in self.feature_results.items():
            model_results = results['model_results']
            scores = [metrics[metric] for metrics in model_results.values() if metrics[metric] > 0]
            avg_scores[n_features] = np.mean(scores) if scores else 0

        best_n = max(avg_scores, key=avg_scores.get)
        return best_n, avg_scores

    def get_feature_comparison_dataframe(self):
        """
        Get results as a pandas DataFrame with feature counts as columns
        """
        # Get all model names
        model_names = list(self.models.keys())

        # Create DataFrame structure
        results_data = {}
        for model_name in model_names:
            results_data[model_name] = {}
            for n_features in sorted(self.feature_results.keys()):
                if model_name in self.feature_results[n_features]['model_results']:
                    results_data[model_name][n_features] = (
                        self.feature_results[n_features]['model_results'][model_name]['f1']
                    )
                else:
                    results_data[model_name][n_features] = 0

        df = pd.DataFrame(results_data).T
        return df

    def plot_feature_comparison(self):
        """
        Create plots comparing performance across different feature counts
        """
        df = self.get_feature_comparison_dataframe()

        # Plot for top 5 models
        avg_performance = df.mean(axis=1).sort_values(ascending=False)
        top_models = avg_performance.head(5).index

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()

        for i, model_name in enumerate(top_models):
            if i < 6:  # Only plot first 6 in 2x3 grid
                axes[i].plot(df.columns, df.loc[model_name], 'o-')
                axes[i].set_title(f'{model_name}')
                axes[i].set_xlabel('Number of Features')
                axes[i].set_ylabel('F1 Score')
                axes[i].grid(True, alpha=0.3)

        # Remove unused subplots
        for i in range(len(top_models), 6):
            fig.delaxes(axes[i])

        plt.tight_layout()
        plt.savefig('feature_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()

        # Overall comparison plot
        plt.figure(figsize=(12, 8))
        for model_name in top_models:
            plt.plot(df.columns, df.loc[model_name], 'o-', label=model_name, linewidth=2)

        plt.xlabel('Number of Features')
        plt.ylabel('F1 Score')
        plt.title('Model Performance vs Number of Features (Top 5 Models)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig('overall_feature_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()

    def detailed_report_best_features(self):
        """
        Generate detailed report for the best feature count configuration
        """
        best_n, avg_scores = self.get_best_feature_count('f1')
        print(f"\\n=== BEST FEATURE COUNT: {best_n} features (avg F1: {avg_scores[best_n]:.4f}) ===")

        # Find best model for this feature count
        model_results = self.feature_results[best_n]['model_results']
        best_model_name = max(model_results.keys(), key=lambda x: model_results[x]['f1'])
        best_model_score = model_results[best_model_name]['f1']

        print(f"Best model: {best_model_name} (F1: {best_model_score:.4f})")
        print(f"Selected features ({len(self.feature_results[best_n]['features'])}):")
        print(self.feature_results[best_n]['features'][:20])  # Show first 20
        if len(self.feature_results[best_n]['features']) > 20:
            print(f"... and {len(self.feature_results[best_n]['features']) - 20} more")

        return best_n, best_model_name

    def save_feature_selection_results(self, output_dir='results'):
        """
        Save detailed results of feature selection
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Save feature comparison dataframe
        df = self.get_feature_comparison_dataframe()
        df.to_csv(f'{output_dir}/feature_comparison_results.csv')

        # Save best features list
        best_n, _ = self.get_best_feature_count('f1')
        best_features = self.feature_results[best_n]['features']
        pd.DataFrame({'feature': best_features}).to_csv(f'{output_dir}/best_features_n{best_n}.csv', index=False)

        # Save detailed results per feature count
        detailed_results = []
        for n_features, results in self.feature_results.items():
            for model_name, metrics in results['model_results'].items():
                row = {'n_features': n_features, 'model': model_name}
                row.update(metrics)
                detailed_results.append(row)

        detailed_df = pd.DataFrame(detailed_results)
        detailed_df.to_csv(f'{output_dir}/detailed_feature_selection_results.csv', index=False)

        print(f"Results saved to {output_dir}/")


# Usage example
def main():
    # Load your data
    df = pd.read_csv("../data/carcinogen_properties.csv")

    # Initialize comparator
    comparator = MLClassifierComparator(random_state=42)

    # Prepare data
    comparator.prepare_data(
        df,
        target_col='Carcinogenicity',
        # You can specify specific features if you want
        # feature_cols=['MolWt', 'LogP', 'TPSA', ...]
    )

    # Compare models across different feature counts
    feature_counts = [25, 50, 100, 150, 'all']
    comparator.compare_models_with_feature_selection(feature_counts=feature_counts)

    # Get best configuration
    best_n, best_model = comparator.detailed_report_best_features()

    # Create visualizations
    comparator.plot_feature_comparison()

    # Save results
    comparator.save_feature_selection_results('results')

    print("\\n=== SUMMARY ===")
    print(f"Best number of features: {best_n}")
    print(f"Best model: {best_model}")


if __name__ == "__main__":
    main()