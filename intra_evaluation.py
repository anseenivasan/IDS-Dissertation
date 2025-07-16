# intra_evaluation.py
import time
import pandas as pd
import numpy as np


# Scikit-learn models
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression

# Scikit-learn preprocessing and model selection
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    balanced_accuracy_score,
    roc_auc_score,
    make_scorer
)
from sklearn.preprocessing import StandardScaler


# Import the custom scorer and the SCORING_METRICS dict
from utilis.metrics_helpers import calculate_specificity, _roc_auc_scorer_wrapper, get_scoring_metrics


# Import constants and helper functions from your utils package
from utilis.constants import (
    DATASET_PATHS, # Not directly used here, but good to know it's available
    PARAM_GRIDS,
    TARGET_ATTACK_LABELS_STR_BROAD, # Use the broad target labels for intra-dataset
    GLOBAL_FLAG_COLUMNS_WITH_NANS, # Not directly used here, but part of data loading context
    BROAD_MAPPING # Used for mapping granular to broad labels
)
from utilis.metrics_helpers import calculate_specificity # Helper for specificity
# Note: label_encoder and ALL_ENCODED_LABELS are passed as arguments from main_evaluation





def run_intra_dataset_evaluation(
    all_combined_dfs,
    label_encoder, # Passed from main_evaluation after fitting on all labels
    ALL_ENCODED_LABELS, # All encoded granular labels
    common_features, # Passed from main_evaluation
    broad_label_mapper, # Function to map granular encoded to broad encoded
    broad_label_encoder, # LabelEncoder for broad labels
    target_report_labels # <-- NEW PARAMETER
):
    """
    Performs intra-dataset generalization evaluation.
    Splits each dataset into train/test, trains models, and evaluates on the test split.

    Args:
        all_combined_dfs (dict): Dictionary of combined DataFrames for each dataset.
        label_encoder (LabelEncoder): Fitted LabelEncoder for consistent label transformation.
        ALL_ENCODED_LABELS (list): List of all possible encoded labels.
        common_features (list): List of features common across all datasets.

    Returns:
        dict: A dictionary containing all evaluation results for intra-dataset scenarios.
    """
    print("\n--- Starting Intra-Dataset Generalization Evaluation ---")

    print("\n--- DEBUG (intra_evaluation): State of broad_label_encoder received ---")
    print(f"DEBUG (intra_evaluation): Type of broad_label_encoder: {type(broad_label_encoder)}")
    if hasattr(broad_label_encoder, 'classes_'):
        print(f"DEBUG (intra_evaluation): broad_label_encoder.classes_: {broad_label_encoder.classes_.tolist()}")
        print(f"DEBUG (intra_evaluation): broad_label_encoder.classes_ length: {len(broad_label_encoder.classes_)}")
    else:
        print(f"DEBUG (intra_evaluation): broad_label_encoder has no 'classes_' attribute.")
    print("-------------------------------------------------------------------------\n")


    # Define base models (will be cloned by GridSearchCV)
    base_models = {
        'Logistic Regression': LogisticRegression(random_state=42, n_jobs=-1),
        'Random Forest': RandomForestClassifier(random_state=42, n_jobs=-1),
        'XGBoost': XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='mlogloss', n_jobs=-1)
    }

    # Cross-validation strategy for HPO (k=5 stratified as per paper)
    cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Scoring metrics for GridSearchCV (as per paper)

    scoring_metrics = get_scoring_metrics(broad_label_encoder) # Corrected line

    refit_metric = 'f1_weighted' # Metric to use for selecting the best model

    all_intra_results = {}

    for dataset_name, combined_df_intra in all_combined_dfs.items():
        print(f"\n################################################################################")
        print(f"### SCENARIO: Intra-Dataset Evaluation for {dataset_name} ###")
        print(f"################################################################################")

        X_intra = combined_df_intra.drop(['Label', 'BroadLabel'], axis=1) # Drop both label columns
        y_intra_granular_encoded = combined_df_intra['Label'] # This is the granular encoded label
        y_intra_broad = combined_df_intra['BroadLabel'] # This is the broad encoded label

        # Impute NaNs introduced by feature alignment with 0 (before train-test split)
        # This handles columns that were missing in a dataset but present in common_features
        print("Imputing NaNs introduced by feature alignment with 0 for intra-dataset features...")
        X_intra.fillna(0, inplace=True)
        print("NaN imputation complete for intra-dataset features.")




        # Perform train-test split within this dataset (e.g., 80/20 split)
        # Split is stratified on the broad labels.
        X_train_intra, X_test_intra, y_train_intra_broad, y_test_intra_broad = train_test_split(
            X_intra, y_intra_broad, test_size=0.2, random_state=42, stratify=y_intra_broad
        )

        
        print(f"X_train_intra shape: {X_train_intra.shape}, y_train_intra_broad shape: {y_train_intra_broad.shape}")
        print(f"X_test_intra shape: {X_test_intra.shape}, y_test_intra_broad shape: {y_test_intra_broad.shape}")

        # Z-score Normalization for this Intra-Dataset Scenario
        print("\nApplying Z-score normalization for this intra-dataset scenario...")
        scaler_intra = StandardScaler()
        numerical_cols_for_scaling_intra = X_train_intra.select_dtypes(include=np.number).columns.tolist()

        scaler_intra.fit(X_train_intra[numerical_cols_for_scaling_intra])

        X_train_intra_scaled = X_train_intra.copy()
        X_test_intra_scaled = X_test_intra.copy()

        X_train_intra_scaled[numerical_cols_for_scaling_intra] = scaler_intra.transform(X_train_intra[numerical_cols_for_scaling_intra])
        X_test_intra_scaled[numerical_cols_for_scaling_intra] = scaler_intra.transform(X_test_intra[numerical_cols_for_scaling_intra])
        print("Z-score normalization complete for intra-dataset scenario.")

        # Create a subsample of the training data for HPO (to manage memory)
        hpo_sample_size_intra = 500000 # Example: 500,000 rows for HPO
        
        if len(X_train_intra_scaled) > hpo_sample_size_intra:
            print(f"\nSubsampling training data for HPO to {hpo_sample_size_intra} rows...")
            X_train_intra_scaled_hpo, _, y_train_intra_broad_hpo , _ = train_test_split(
                X_train_intra_scaled, y_train_intra_broad,
                train_size=hpo_sample_size_intra, stratify=y_train_intra_broad, random_state=42
            )
            print(f"HPO subsample shape: {X_train_intra_scaled_hpo.shape}")
        else:
            X_train_intra_scaled_hpo = X_train_intra_scaled
            y_train_intra_broad_hpo = y_train_intra_broad
            print("Training data size is small enough, no subsampling for HPO.")

        intra_dataset_results = {} # Store results for current intra-dataset

        # Hyperparameter Optimization and Model Training/Evaluation
        for model_name, base_model in base_models.items():
            print(f"\n--- Hyperparameter Optimization for {model_name} (Intra-Dataset: {dataset_name}) ---")
            param_grid = PARAM_GRIDS[model_name] # Use PARAM_GRIDS from constants

            search_start_time = time.time()

            # GridSearchCV for all models as per paper
            search = GridSearchCV(
                estimator=base_model,
                param_grid=param_grid,
                scoring=scoring_metrics,
                refit=refit_metric,
                cv=cv_strategy,
                n_jobs=-1, # Use all available cores for HPO (careful with RAM)
                verbose=2 # Set to 0 for less output during search
            )
            
            # Fit GridSearchCV on the HPO subsample
            search.fit(X_train_intra_scaled_hpo, y_train_intra_broad_hpo)

            search_end_time = time.time()
            search_duration = search_end_time - search_start_time

            print(f"Hyperparameter optimization for {model_name} completed in {search_duration:.2f} seconds.")
            print(f"Best parameters for {model_name}: {search.best_params_}")
            print(f"Best cross-validation score ({refit_metric}) for {model_name}: {search.best_score_:.4f}")

            # Store mean CV metrics as per paper
            cv_means = {}
            # Iterate over the keys of the scoring_metrics dictionary to get all expected metrics
        for metric_key in scoring_metrics.keys(): # Use metric_key instead of metric_name
            # Construct the full key for cv_results_
            cv_result_key = f'mean_test_{metric_key}'
            
            # --- MODIFIED: Check if the key exists in search.cv_results_ ---
            if cv_result_key in search.cv_results_:
                # Retrieve the mean score for the best parameters
                cv_means[metric_key] = search.cv_results_[cv_result_key][search.best_index_]
                print(f"  Mean CV {metric_key}: {cv_means[metric_key]:.4f}")
            else:
                # If the key doesn't exist (e.g., due to scoring failure in all folds for that metric)
                cv_means[metric_key] = np.nan # Assign NaN
                print(f"  Mean CV {metric_key}: N/A (Scoring failed or key not found)")

            # Get the best model found by GridSearchCV
            best_model = search.best_estimator_

            # --- Evaluation on Intra-Dataset Test Set ---
            print(f"\n--- Evaluation of Optimized {model_name} on Intra-Dataset Test Set ({dataset_name}) ---")
            
            # Make predictions and get probabilities
            y_pred_intra = best_model.predict(X_test_intra_scaled)
            y_proba_intra = best_model.predict_proba(X_test_intra_scaled)

            # --- THIS IS THE EXACT LOCATION FOR PADDING y_proba_combined ---
            # It's placed here because y_proba_combined is just created.
            
            # --- NEW: Pad y_proba_intra to match ALL_ENCODED_LABELS size ---
            padded_y_proba_intra = np.zeros((y_proba_intra.shape[0], len(broad_label_encoder.classes_)))
            for i, model_predicted_int_class in enumerate(best_model.classes_):
                if 0 <= model_predicted_int_class < len(broad_label_encoder.classes_):
                    padded_y_proba_intra[:, model_predicted_int_class] = y_proba_intra[:, i]
                else:
                    print(f"Warning: Model predicted class {model_predicted_int_class} is out of range for broad_label_encoder.classes_ (0 to {len(broad_label_encoder.classes_)-1}). Skipping padding for this class.")
             
            # --- END OF PADDING ---

             

            # Get labels present in this specific test set for reporting
            
            current_test_labels_encoded_intra = np.sort(y_test_intra_broad.unique())
            current_test_labels_decoded_intra = broad_label_encoder.inverse_transform(current_test_labels_encoded_intra)
            
            
            # Calculate overall metrics
            intra_accuracy = accuracy_score(y_test_intra_broad, y_pred_intra)
            intra_precision = precision_score(y_test_intra_broad, y_pred_intra, average='weighted', zero_division=0)
            intra_recall = recall_score(y_test_intra_broad, y_pred_intra, average='weighted', zero_division=0)
            intra_f1 = f1_score(y_test_intra_broad, y_pred_intra, average='weighted', zero_division=0)
            intra_balanced_accuracy = balanced_accuracy_score(y_test_intra_broad, y_pred_intra)

            # Corrected: Pass broad_label_encoder object to calculate_specificity
            intra_specificity_scores = calculate_specificity(y_test_intra_broad, y_pred_intra, broad_label_encoder)
            intra_specificity = np.mean(list(intra_specificity_scores.values()))


            

            # --- MODIFIED: Use padded_y_proba_intra for ROC AUC calculation ---
            intra_roc_auc = np.nan # Default to NaN if calculation fails
            try:
                if len(np.unique(y_test_intra_broad)) > 1:
                    intra_roc_auc = roc_auc_score(
                        y_test_intra_broad,
                        padded_y_proba_intra, # Use the padded probabilities
                        multi_class='ovr',
                        average='weighted',
                        labels=np.arange(len(broad_label_encoder.classes_)) # Pass integer labels for ROC AUC
                    )
                else:
                    print(f"  Warning: Intra-dataset ROC AUC undefined (only one class in true labels for {dataset_name}).")
            except ValueError as e:
                print(f"  Warning: Could not calculate intra-dataset ROC AUC for {dataset_name}. Error: {e}")
            except Exception as e:
                print(f"  Warning: Unexpected error calculating intra-dataset ROC AUC for {dataset_name}. Error: {e}")
            # --- END MODIFIED ---
            

            print(f"Accuracy: {intra_accuracy:.4f}")
            print(f"Precision (weighted): {intra_precision:.4f}")
            print(f"Recall (weighted): {intra_recall:.4f}")
            print(f"F1-Score (weighted): {intra_f1:.4f}")
            print(f"Balanced Accuracy: {intra_balanced_accuracy:.4f}")
            print(f"Specificity (avg): {intra_specificity:.4f}")
            print(f"ROC AUC (weighted): {intra_roc_auc:.4f}")

            # Extract specific attack class metrics
            # In intra_evaluation.py (intra-dataset metrics)
            intra_clf_report_dict = classification_report(
                y_test_intra_broad,
                y_pred_intra,
                output_dict=True,
                zero_division=0,
                labels=np.arange(len(broad_label_encoder.classes_)), # <-- ADD THIS LINE
                target_names=broad_label_encoder.classes_ # <-- ADD THIS LINE
            )
            specific_intra_metrics = {}
            for target_str_label in target_report_labels:
                # The keys in intra_clf_report_dict will be the string labels if target_names was used in classification_report,
                # or integer labels if not. Since we want to use TARGET_ATTACK_LABELS_STR_BROAD (strings),
                # we need to ensure the report keys match.
                # classification_report with output_dict=True uses string labels if target_names is provided,
                # otherwise it uses the integer labels.
                # Let's ensure target_names is used for the report.
                # For now, assume report keys are strings.
                if target_str_label in intra_clf_report_dict:
                    specific_intra_metrics[target_str_label] = {
                        'Precision': intra_clf_report_dict[target_str_label]['precision'],
                        'Recall': intra_clf_report_dict[target_str_label]['recall'],
                        'F1-Score': intra_clf_report_dict[target_str_label]['f1-score'],
                        'Support': intra_clf_report_dict[target_str_label]['support']
                    }
                else:
                    specific_intra_metrics[target_str_label] = {
                        'Precision': None, 'Recall': None, 'F1-Score': None, 'Support': 0
                    }
                    print(f"  Warning: Class '{target_str_label}' not found in intra-dataset test set for '{dataset_name}'.")
            
            # Store all results for the current model in the current intra-dataset scenario
            intra_dataset_results[model_name] = {
                'Best Params': search.best_params_,
                'Optimization Time (s)': search_duration,
                'CV Metrics': cv_means,
                'Intra-Dataset Test Metrics': {
                    'Accuracy': intra_accuracy,
                    'Precision': intra_precision,
                    'Recall': intra_recall,
                    'F1-Score': intra_f1,
                    'Balanced Accuracy': intra_balanced_accuracy,
                    'Specificity': intra_specificity,
                    'ROC AUC': intra_roc_auc,
                    'Specific Attack Metrics': specific_intra_metrics
                }
            }
    # Store results for the current intra-dataset scenario
    all_intra_results[dataset_name] = intra_dataset_results

    return all_intra_results # Return the collected results

