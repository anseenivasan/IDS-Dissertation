# cross_evaluation.py
import time
import pandas as pd
import numpy as np



# Scikit-learn models
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
 


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

# Import from your utils module
from utilis.Data_loader import load_and_align_all_data # This loads all data and performs initial alignment/encoding
from utilis.metrics_helpers import calculate_specificity,_roc_auc_scorer_wrapper, get_scoring_metrics
from utilis.constants import (
    DATASET_PATHS, 
    PARAM_GRIDS, 
    TARGET_ATTACK_LABELS_STR_BROAD, 
    BROAD_MAPPING # Used for mapping granular to broad labels
)


 

def run_cross_dataset_evaluation(all_combined_dfs, all_individual_dfs_by_dataset, label_encoder, ALL_ENCODED_LABELS, common_features,broad_label_mapper,broad_label_encoder,target_report_labels):
    """
    Performs cross-dataset generalization evaluation.
    Trains models on one dataset and tests on another, including hyperparameter optimization
    and per-day evaluation on the test set.

    Args:
        all_combined_dfs (dict): Dictionary of combined DataFrames for each dataset (granular labels).
        all_individual_dfs_by_dataset (dict): Dictionary of individual daily DataFrames for each dataset (granular labels).
        label_encoder (LabelEncoder): Fitted LabelEncoder for granular labels.
        ALL_ENCODED_LABELS (list): List of all possible encoded granular labels.
        common_features (list): List of features common across all datasets.
        broad_label_mapper (function): Function to map granular encoded labels to broad encoded labels.
        broad_label_encoder (LabelEncoder): Fitted LabelEncoder for broad labels.

    Returns:
        dict: A dictionary containing all evaluation results for cross-dataset scenarios.
    """

    print("\n--- Starting Cross-Dataset Generalization Evaluation ---")

    print("\n--- DEBUG (cross_evaluation): State of broad_label_encoder received ---")
    print(f"DEBUG (cross_evaluation): Type of broad_label_encoder: {type(broad_label_encoder)}")
    if hasattr(broad_label_encoder, 'classes_'):
        print(f"DEBUG (cross_evaluation): broad_label_encoder.classes_: {broad_label_encoder.classes_.tolist()}")
        print(f"DEBUG (cross_evaluation): broad_label_encoder.classes_ length: {len(broad_label_encoder.classes_)}")
    else:
        print(f"DEBUG (cross_evaluation): broad_label_encoder has no 'classes_' attribute.")
    print("-------------------------------------------------------------------------\n")


    SCENARIOS = [
        ('CIC_IDS_2018', 'CIC_IDS_2017'),
        ('CIC_IDS_2017', 'CIC_IDS_2018')
    ]

    base_models = {
        'Logistic Regression': LogisticRegression(random_state=42, n_jobs=-1),
        'Random Forest': RandomForestClassifier(random_state=42, n_jobs=-1),
        'XGBoost': XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='mlogloss', n_jobs=-1)
    }
    
    cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Use the SCORING_METRICS dictionary defined in utils/metrics_helpers.py
    # We need to dynamically set the 'labels' argument for the roc_auc_weighted scorer.

    scoring_metrics = get_scoring_metrics(broad_label_encoder)
    refit_metric = 'f1_weighted'

    all_scenario_results = {}

    for train_name, test_name in SCENARIOS:
        print(f"\n################################################################################")
        print(f"### SCENARIO: Training on {train_name} and Testing on {test_name} ###")
        print(f"################################################################################")

        # Prepare training and testing data for the current scenario (still granular labels)
        X_train_scenario = all_combined_dfs[train_name].drop(['Label', 'BroadLabel'], axis=1)
        X_test_scenario_combined = all_combined_dfs[test_name].drop(['Label', 'BroadLabel'], axis=1)
        
        
        # y_train_scenario_broad and y_test_scenario_combined_broad are already prepared in data_loader
        y_train_scenario_broad = all_combined_dfs[train_name]['BroadLabel']
        y_test_scenario_combined_broad = all_combined_dfs[test_name]['BroadLabel']

        

        print(f"X_train_scenario shape: {X_train_scenario.shape}, y_train_scenario_broad shape: {y_train_scenario_broad.shape}")
        print(f"X_test_scenario_combined shape: {X_test_scenario_combined.shape}, y_test_scenario_combined_broad shape: {y_test_scenario_combined_broad.shape}")

        # Impute NaNs introduced by feature alignment with 0 (before scaling)
        # This handles columns that were missing in a dataset but present in common_features
        print("\nImputing NaNs introduced by feature alignment with 0...")
        X_train_scenario.fillna(0, inplace=True)
        X_test_scenario_combined.fillna(0, inplace=True)
        print("NaN imputation complete for scenario features.")

        # Z-score Normalization
        print("\nApplying Z-score normalization for this scenario...")
        scaler = StandardScaler()
        numerical_cols_for_scaling = X_train_scenario.select_dtypes(include=np.number).columns.tolist()

        scaler.fit(X_train_scenario[numerical_cols_for_scaling])

        X_train_scenario_scaled = X_train_scenario.copy()
        X_train_scenario_scaled[numerical_cols_for_scaling] = scaler.transform(X_train_scenario[numerical_cols_for_scaling])
        print("Z-score normalization complete for scenario.")

        # Create a subsample of the training data for HPO (to manage memory)
        hpo_sample_size = 500000 # Example: 500,000 rows for HPO
        
        if len(X_train_scenario_scaled) > hpo_sample_size:
            print(f"\nSubsampling training data for HPO to {hpo_sample_size} rows...")
            X_train_scenario_scaled_hpo, _, y_train_scenario_broad_hpo, _ = train_test_split(
                X_train_scenario_scaled, y_train_scenario_broad, # Use broad labels for HPO subsample
                train_size=hpo_sample_size, stratify=y_train_scenario_broad, random_state=42
            )
            print(f"HPO subsample shape: {X_train_scenario_scaled_hpo.shape}")
        else:
            X_train_scenario_scaled_hpo = X_train_scenario_scaled
            y_train_scenario_broad_hpo = y_train_scenario_broad
            print("Training data size is small enough, no subsampling for HPO.")

        scenario_results = {}

        # Hyperparameter Optimization and Model Training/Evaluation
        for model_name, base_model in base_models.items():
            print(f"\n--- Hyperparameter Optimization for {model_name} (Training on {train_name}) ---")
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
            
            # Fit GridSearchCV on the HPO subsample (with broad labels)
            search.fit(X_train_scenario_scaled_hpo, y_train_scenario_broad_hpo)

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

            # --- Overall Evaluation on Combined Test Set ---
            print(f"\n--- Overall Evaluation of Optimized {model_name} on Combined Test Set ({test_name}) ---")
            # Scale the combined test set using the scaler fitted on the training data
            X_test_scenario_combined_scaled = X_test_scenario_combined.copy()
            X_test_scenario_combined_scaled[numerical_cols_for_scaling] = scaler.transform(X_test_scenario_combined[numerical_cols_for_scaling])
            
            # Make predictions and get probabilities
            y_pred_combined = best_model.predict(X_test_scenario_combined_scaled)
            y_proba_combined = best_model.predict_proba(X_test_scenario_combined_scaled)


            # Pad y_proba_combined to match broad_label_encoder.classes_ size
            padded_y_proba_combined = np.zeros((y_proba_combined.shape[0], len(broad_label_encoder.classes_)))
            # best_model.classes_ contains the integer labels the model was trained on.
             #y_proba_combined[:, i] corresponds to the probability for best_model.classes_[i]
            for i, model_predicted_int_class in enumerate(best_model.classes_):
                # The model_predicted_int_class is already the correct integer index
                # for the broad_label_encoder's class array.
                
                # Check if this integer is within the valid range of indices
                # for the broad_label_encoder's classes. This is a robust check.
                if 0 <= model_predicted_int_class < len(broad_label_encoder.classes_):
                    
                    padded_y_proba_combined[:, model_predicted_int_class] = y_proba_combined[:, i]
                else:

                    # This warning should now only appear if there's a serious mismatch
                    # (e.g., model predicted class 10, but broad_label_encoder only has 9 classes 0-8)
                    print(f"Warning: Model predicted class {model_predicted_int_class} is out of range for broad_label_encoder.classes_ (0 to {len(broad_label_encoder.classes_)-1}). Skipping padding for this class.")


            # Get labels present in this specific test set for reporting
            current_test_labels_encoded = np.unique(y_test_scenario_combined_broad)

            current_test_labels_decoded = broad_label_encoder.inverse_transform(current_test_labels_encoded)
            

            # Calculate overall metrics
            overall_accuracy = accuracy_score(y_test_scenario_combined_broad, y_pred_combined)
            overall_precision = precision_score(y_test_scenario_combined_broad, y_pred_combined, average='weighted', zero_division=0)
            overall_recall = recall_score(y_test_scenario_combined_broad, y_pred_combined, average='weighted', zero_division=0)
            overall_f1 = f1_score(y_test_scenario_combined_broad, y_pred_combined, average='weighted', zero_division=0)
            overall_balanced_accuracy = balanced_accuracy_score(y_test_scenario_combined_broad, y_pred_combined)
            
            # Corrected: Pass broad_label_encoder object to calculate_specificity
            overall_specificity_scores = calculate_specificity(y_test_scenario_combined_broad, y_pred_combined, broad_label_encoder)
            overall_specificity = np.mean(list(overall_specificity_scores.values()))

                     # --- MODIFIED: Robust ROC AUC calculation for overall ---
            overall_roc_auc = np.nan # Default to NaN if calculation fails
            try:
                if len(np.unique(y_test_scenario_combined_broad)) > 1:
                    overall_roc_auc = roc_auc_score(
                        y_test_scenario_combined_broad,
                        padded_y_proba_combined, # Use the padded probabilities
                        multi_class='ovr',
                        average='weighted',
                        labels=np.arange(len(broad_label_encoder.classes_)) # Pass integer labels for ROC AUC  
                    )
                else:
                    print(f"  Warning: Only one class in y_test_scenario_combined. ROC AUC undefined.")
            except ValueError as e:
                print(f"  Warning: Could not calculate overall ROC AUC. Error: {e}")
                overall_roc_auc = np.nan # Assign NaN on error

            except Exception as e: # Catch any other unexpected errors
                print(f"  Warning: Unexpected error calculating overall ROC AUC. Error: {e}")
                


            print(f"Overall Accuracy: {overall_accuracy:.4f}")
            print(f"Overall Precision (weighted): {overall_precision:.4f}")
            print(f"Overall Recall (weighted): {overall_recall:.4f}")
            print(f"Overall F1-Score (weighted): {overall_f1:.4f}")
            print(f"Overall Balanced Accuracy: {overall_balanced_accuracy:.4f}")
            print(f"Overall Specificity (avg): {overall_specificity:.4f}")
            print(f"Overall ROC AUC (weighted): {overall_roc_auc:.4f}")

            # Extract specific attack class metrics for overall test set
            
            overall_clf_report_dict = classification_report(
                y_test_scenario_combined_broad,
                y_pred_combined,
                output_dict=True,
                zero_division=0,
                labels=np.arange(len(broad_label_encoder.classes_)), # <-- ADD THIS LINE
                target_names=broad_label_encoder.classes_
            )
            specific_overall_metrics = {}
            for target_str_label in target_report_labels: # Use broad target labels
                if target_str_label in overall_clf_report_dict:
                    specific_overall_metrics[target_str_label] = {
                        'Precision': overall_clf_report_dict[target_str_label]['precision'],
                        'Recall': overall_clf_report_dict[target_str_label]['recall'],
                        'F1-Score': overall_clf_report_dict[target_str_label]['f1-score'],
                        'Support': overall_clf_report_dict[target_str_label]['support']
                    }
                else:
                    specific_overall_metrics[target_str_label] = {
                        'Precision': None, 'Recall': None, 'F1-Score': None, 'Support': 0
                    }
                    print(f"  Warning: Class '{target_str_label}' not found in combined test set '{test_name}'.")
            
            # Store all results for the current model in the current scenario
            scenario_results[model_name] = {
                'Best Params': search.best_params_,
                'Optimization Time (s)': search_duration,
                'CV Metrics': cv_means,
                'Overall Metrics (Combined Test Set)': {
                    'Accuracy': overall_accuracy,
                    'Precision': overall_precision,
                    'Recall': overall_recall,
                    'F1-Score': overall_f1,
                    'Balanced Accuracy': overall_balanced_accuracy,
                    'Specificity': overall_specificity,
                    'ROC AUC': overall_roc_auc,
                    'Specific Attack Metrics': specific_overall_metrics
                },
                'Per_Day_Metrics': {} # Initialize for per-day results
            }

            # --- Per-Day Evaluation on Test Set ---
            print(f"\n--- Per-Day Evaluation of Optimized {model_name} on Test Set ({test_name}) ---")
            individual_test_dfs = all_individual_dfs_by_dataset[test_name]
            
            for day_name, day_df in individual_test_dfs.items():
                print(f"\nEvaluating on Day: {day_name}")
                X_day = day_df.drop(['Label', 'BroadLabel'], axis=1) # Drop both label columns
                y_day_broad = day_df['BroadLabel'] # Use the broad encoded label

                # Ensure X_day has common features and is encoded
                X_day = X_day[common_features]
                # Fill NaNs introduced by align_to_common_features for this day
                X_day.fillna(0, inplace=True) 
                
                # Scale the individual day's data using the SAME scaler fitted on X_train_scenario
                X_day_scaled = X_day.copy()
                X_day_scaled[numerical_cols_for_scaling] = scaler.transform(X_day[numerical_cols_for_scaling])

                y_pred_day = best_model.predict(X_day_scaled)
                y_proba_day = best_model.predict_proba(X_day_scaled)

                

            # Pad y_proba_day to match broad_label_encoder.classes_ size
                padded_y_proba_day = np.zeros((y_proba_day.shape[0], len(broad_label_encoder.classes_)))
                for i, model_predicted_int_class in enumerate(best_model.classes_):
                    if 0 <= model_predicted_int_class < len(broad_label_encoder.classes_):
                        padded_y_proba_day[:, model_predicted_int_class] = y_proba_day[:, i]
                    else:
                        print(f"Warning: Model predicted class {model_predicted_int_class} is out of range for broad_label_encoder.classes_ (0 to {len(broad_label_encoder.classes_)-1}). Skipping padding for this class.")




                
                # Get labels present in this specific test day for reporting (broad labels)
                current_day_labels_encoded = np.sort(y_day_broad.unique())
                current_day_labels_decoded = broad_label_encoder.inverse_transform(current_day_labels_encoded)
                

                # Calculate per-day metrics
                day_accuracy = accuracy_score(y_day_broad, y_pred_day)
                day_precision = precision_score(y_day_broad, y_pred_day, average='weighted', zero_division=0)
                day_recall = recall_score(y_day_broad, y_pred_day, average='weighted', zero_division=0)
                day_f1 = f1_score(y_day_broad, y_pred_day, average='weighted', zero_division=0)
                day_balanced_accuracy = balanced_accuracy_score(y_day_broad, y_pred_day)
                # Corrected: Pass broad_label_encoder object to calculate_specificity
                day_specificity_scores = calculate_specificity(y_day_broad, y_pred_day, broad_label_encoder)
                day_specificity = np.mean(list(day_specificity_scores.values()))

                
                   # --- MODIFIED BLOCK FOR day_roc_auc ---
                day_roc_auc = np.nan # Default to NaN if calculation fails
                try:
                    if len(np.unique(y_day_broad)) > 1:
                        day_roc_auc = roc_auc_score(
                            y_day_broad,
                            padded_y_proba_day, # Use the padded probabilities
                            multi_class='ovr',
                            average='weighted',
                            labels=np.arange(len(broad_label_encoder.classes_)) # Use broad encoded classes as reference
                        )
                    else:
                        print(f"  Warning: ROC AUC undefined for {day_name} (only one class in true labels).")
                except ValueError as e:
                    print(f"  Warning: Could not calculate ROC AUC for {day_name}. Error: {e}")
                except Exception as e:
                    print(f"  Warning: Unexpected error calculating ROC AUC for {day_name}. Error: {e}")
 
                print(f"  Accuracy: {day_accuracy:.4f}")
                print(f"  Precision (weighted): {day_precision:.4f}")
                print(f"  Recall (weighted): {day_recall:.4f}")
                print(f"  F1-Score (weighted): {day_f1:.4f}")
                print(f"  Balanced Accuracy: {day_balanced_accuracy:.4f}")
                print(f"  Specificity (avg): {day_specificity:.4f}")
                print(f"  ROC AUC (weighted): {day_roc_auc:.4f}")
                
                print(f"\n  Per-Class Report for {day_name}:")
                day_clf_report_dict = classification_report(
                    y_day_broad,
                    y_pred_day,
                    output_dict=True,
                    zero_division=0,
                    labels=np.arange(len(broad_label_encoder.classes_)), # <-- ADD THIS LINE
                    target_names=broad_label_encoder.classes_ # <-- ADD THIS LINE
                )
                print(classification_report(
                    y_day_broad,
                    y_pred_day,
                    zero_division=0,
                    labels=np.arange(len(broad_label_encoder.classes_)), # <-- ADD THIS LINE
                    target_names=broad_label_encoder.classes_ # <-- USE THE FULL LIST HERE
                ))
                
                # Extract specific attack class metrics for this day
                specific_day_metrics = {}
                for target_str_label in target_report_labels:
                    if target_str_label in day_clf_report_dict:
                        specific_day_metrics[target_str_label] = {
                            'Precision': day_clf_report_dict[target_str_label]['precision'],
                            'Recall': day_clf_report_dict[target_str_label]['recall'],
                            'F1-Score': day_clf_report_dict[target_str_label]['f1-score'],
                            'Support': day_clf_report_dict[target_str_label]['support']
                        }
                    else:
                        specific_day_metrics[target_str_label] = {
                            'Precision': None, 'Recall': None, 'F1-Score': None, 'Support': 0
                        }
                        print(f"    Warning: Class '{target_str_label}' not found in day '{day_name}'.")

                # Store per-day results
                scenario_results[model_name]['Per_Day_Metrics'][day_name] = {
                    'Accuracy': day_accuracy,
                    'Precision': day_precision,
                    'Recall': day_recall,
                    'F1-Score': day_f1,
                    'Balanced Accuracy': day_balanced_accuracy,
                    'Specificity': day_specificity,
                    'ROC AUC': day_roc_auc,
                    'Specific Attack Metrics': specific_day_metrics
                }
            print("-" * 60 + "\n")
        
        # Store results for the current scenario
        all_scenario_results[f"Train_{train_name}_Test_{test_name}"] = scenario_results

    return all_scenario_results # Return the collected results


