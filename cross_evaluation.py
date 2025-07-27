# cross_evaluation.py
import time
import pandas as pd
import numpy as np
import gc


# NEW IMPORTS FOR SAVING/LOADING
import joblib
import os

# Scikit-learn models
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
 


from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split, RandomizedSearchCV
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
from sklearn.utils.class_weight import compute_class_weight # NEW: Import for class weights



# Import from   utils module
from utilis.Data_loader import load_and_align_all_data # This loads all data and performs initial alignment/encoding
from utilis.metrics_helpers import calculate_specificity,_roc_auc_scorer_wrapper, get_scoring_metrics
from utilis.constants import (
    DATASET_PATHS, 
    PARAM_GRIDS, 
    TARGET_ATTACK_LABELS_STR_BROAD, 
    BROAD_MAPPING # Used for mapping granular to broad labels
)


# NEW CONSTANTS FOR SAVING/LOADING
# It's good practice to have separate directories for cross-evaluation models/results
CROSS_MODELS_DIR = 'saved_cross_models'
CROSS_RESULTS_DIR = 'saved_cross_results'

def downsample_benign_data(X, y, broad_label_encoder, ratio=1.0, random_state=42):
    """
    Downsamples the 'Benign' class in the training data.
    Args:
        X (np.ndarray): Features (scaled, potentially PCA-transformed).
        y (pd.Series or np.ndarray): Broad labels.
        broad_label_encoder (LabelEncoder): Fitted LabelEncoder for broad labels.
        ratio (float): Ratio of benign samples to total attack samples (e.g., 1.0 for 1:1).
        random_state (int): Random state for reproducibility.
    Returns:
        tuple: (X_resampled, y_resampled)
    """
    print(f"  Applying Benign downsampling with ratio: {ratio}...")
    
    # Ensure y is a Pandas Series for easy filtering and sampling
    if not isinstance(y, pd.Series):
        y_series = pd.Series(y)
    else:
        y_series = y.copy() # Use .copy() to avoid SettingWithCopyWarning if y was a view
    
    # CRITICAL FIX: Reset the index of y_series to align with the new DataFrame X_df
    y_series = y_series.reset_index(drop=True) # <--- ADD THIS LINE

    benign_encoded_id = broad_label_encoder.transform(['Benign'])[0]

    print(f"  DEBUG (downsample): Benign encoded ID: {benign_encoded_id}")
    print(f"  DEBUG (downsample): y_series dtype: {y_series.dtype}")
    print(f"  DEBUG (downsample): y_series value_counts BEFORE filtering:\n{y_series.value_counts()}")

    # Convert X to DataFrame temporarily to keep labels aligned during filtering/sampling
    X_df = pd.DataFrame(X)
    X_df['y'] = y_series # This assignment should now be perfectly aligned by position

    attack_df = X_df[X_df['y'] != benign_encoded_id]
    benign_df = X_df[X_df['y'] == benign_encoded_id]

    print(f"  DEBUG (downsample): Length of attack_df after filtering: {len(attack_df)}")
    print(f"  DEBUG (downsample): Length of benign_df after filtering: {len(benign_df)}")

    # Calculate sample size for benign data
    if len(attack_df) == 0:
        print("  No attack samples found for downsampling. Returning original data.")
        return X, y
    
    sample_size = min(len(benign_df), int(len(attack_df) * ratio))

    if sample_size == 0:
        print("  Calculated benign sample size is 0. Returning only attack data.")
        return attack_df.drop(columns=['y']).values, attack_df['y'].values

    benign_sample = benign_df.sample(n=sample_size, random_state=random_state)

    # Concatenate attack and sampled benign data
    resampled_df = pd.concat([attack_df, benign_sample], ignore_index=True)

    # Separate X and y, convert X back to NumPy array
    X_resampled = resampled_df.drop(columns=['y']).values
    y_resampled = resampled_df['y'].values

    print(f"  Original data shape: {X.shape}")
    print(f"  Attack samples: {len(attack_df)}, Benign samples (original): {len(benign_df)}")
    print(f"  Benign samples (resampled): {len(benign_sample)}")
    print(f"  Resampled data shape: {X_resampled.shape}")
    print(f"  Resampled label distribution:\n{pd.Series(y_resampled).value_counts()}")
    
    gc.collect()
    return X_resampled, y_resampled
 

def run_cross_dataset_evaluation(all_combined_dfs, all_individual_dfs_by_dataset, label_encoder, ALL_ENCODED_LABELS, common_features,broad_label_mapper,broad_label_encoder,target_report_labels,force_retrain: bool = False):
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
    print(f"DEBUG (cross_evaluation): broad_label_encoder.classes_: {broad_label_encoder.classes_.tolist()}")
      
    print("-------------------------------------------------------------------------\n")


    SCENARIOS = [
        ('CIC_IDS_2018', 'CIC_IDS_2017'),
        ('CIC_IDS_2017', 'CIC_IDS_2018')
    ]

    base_models = {
       # 'Logistic Regression': LogisticRegression(random_state=42, n_jobs=-1),
        #'Random Forest': RandomForestClassifier(random_state=42, n_jobs=-1),
        'XGBoost': XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='mlogloss', n_jobs=-1)
    }
    
    cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Use the SCORING_METRICS dictionary defined in utils/metrics_helpers.py
    scoring_metrics = get_scoring_metrics(broad_label_encoder)
    refit_metric = 'f1_weighted_attack_only'

    all_scenario_results = {}


    # NEW: Pre-calculate integer labels for attack classes for F1-Score (Attack Only)
    attack_labels_string = [label for label in target_report_labels if label != 'Benign']
    attack_labels_encoded = broad_label_encoder.transform(attack_labels_string)


    for train_name, test_name in SCENARIOS:
        print(f"\n################################################################################")
        print(f"### SCENARIO: Training on {train_name} and Testing on {test_name} ###")
        print(f"################################################################################")

        # Prepare training and testing data for the current scenario  
        X_train_scenario_df = all_combined_dfs[train_name].drop(['Label', 'BroadLabel'], axis=1)
        y_train_scenario_broad = all_combined_dfs[train_name]['BroadLabel']
        
        
        X_test_scenario_combined_df = all_combined_dfs[test_name].drop(['Label', 'BroadLabel'], axis=1)
        y_test_scenario_combined_broad = all_combined_dfs[test_name]['BroadLabel']

        # NEW: Convert numerical columns to float32 (if not already) and ensure consistent dtypes
        # This part is included for robustness, even if parquet already optimizes
        # print("Converting numerical columns to float32 for scenario data...")
        # for df in [X_train_scenario_df, X_test_scenario_combined_df]:
        #     for col in df.select_dtypes(include=np.number).columns:
        #         df[col] = df[col].astype(np.float32)
        # gc.collect()

        print(f"X_train_scenario shape: {X_train_scenario_df.shape}, y_train_scenario_broad shape: {y_train_scenario_broad.shape}")
        print(f"X_test_scenario_combined shape: {X_test_scenario_combined_df.shape}, y_test_scenario_combined_broad shape: {y_test_scenario_combined_broad.shape}")

        # Impute NaNs introduced by feature alignment with 0 (before scaling)
        # This handles columns that were missing in a dataset but present in common_features
        print("\nImputing NaNs introduced by feature alignment with 0...")
        X_train_scenario_df.fillna(0, inplace=True)
        X_test_scenario_combined_df.fillna(0, inplace=True)
        print("NaN imputation complete for scenario features.")

        # Z-score Normalization
        print("\nApplying Z-score normalization for this scenario...")
        scaler = StandardScaler()
        numerical_cols_for_scaling = X_train_scenario_df.select_dtypes(include=np.number).columns.tolist()

        scaler.fit(X_train_scenario_df[numerical_cols_for_scaling])

        X_train_scenario_scaled = X_train_scenario_df.copy()
        X_train_scenario_scaled[numerical_cols_for_scaling] = scaler.transform(X_train_scenario_df[numerical_cols_for_scaling])
        

        # Convert to NumPy array for consistent input to models
        X_train_scenario_scaled_np = X_train_scenario_scaled.values

        # Free up memory from original DataFrames
        del X_train_scenario_df, X_train_scenario_scaled
        gc.collect()

        print("Z-score normalization complete for scenario.")

        print(f"DEBUG: y_train_scenario_broad value_counts for {train_name} (full training set, before HPO subsampling):\n{y_train_scenario_broad.value_counts()}")
        print(f"DEBUG: y_train_scenario_broad unique values for {train_name}: {np.unique(y_train_scenario_broad).tolist()}")



        # Create a subsample of the training data for HPO (to manage memory)
        hpo_sample_size = 1000000 # Example: 500,000 rows for HPO

        X_train_scenario_hpo_source = X_train_scenario_scaled_np # Source for HPO subsample
        y_train_scenario_hpo_source = y_train_scenario_broad # Source for HPO subsample

        # NEW DEBUG PRINT: Directly before the train_test_split call
        print(f"\nDEBUG: y_hpo_base value_counts IMMEDIATELY BEFORE HPO subsampling train_test_split:")
        print(y_train_scenario_hpo_source.value_counts())
        print(f"DEBUG: y_hpo_base unique values IMMEDIATELY BEFORE HPO subsampling train_test_split: {np.unique(y_train_scenario_hpo_source).tolist()}")
        print(f"DEBUG: y_hpo_base total length: {len(y_train_scenario_hpo_source)}")
        print(f"DEBUG: Requested hpo_sample_size: {hpo_sample_size}")
        # END NEW DEBUG PRINT
        
        if len(X_train_scenario_hpo_source) > hpo_sample_size:
            print(f"\nSubsampling training data for HPO to {hpo_sample_size} rows...")
            X_train_scenario_scaled_hpo, _, y_train_scenario_broad_hpo, _ = train_test_split(
                X_train_scenario_hpo_source, y_train_scenario_hpo_source, # Use broad labels for HPO subsample
                train_size=hpo_sample_size, stratify=y_train_scenario_hpo_source, random_state=42
            )
            print(f"DEBUG: y_train_scenario_broad_hpo distribution *AFTER* train_test_split:")
            print(pd.Series(y_train_scenario_broad_hpo).value_counts())
            print(f"DEBUG: y_train_scenario_broad_hpo unique values *AFTER* train_test_split: {np.unique(y_train_scenario_broad_hpo).tolist()}")
            print(f"DEBUG: y_train_scenario_broad_hpo length *AFTER* train_test_split: {len(y_train_scenario_broad_hpo)}")
            
        else:
            X_train_scenario_scaled_hpo = X_train_scenario_hpo_source
            y_train_scenario_broad_hpo = y_train_scenario_hpo_source
            print("Training data size is small enough, no subsampling for HPO.")

        # NEW: Apply Benign Downsampling to the HPO subsample
        DOWNSAMPLE_BENIGN_RATIO_HPO = 1.0 # Example: 1:1 ratio of attack to benign for HPO
        X_train_scenario_scaled_hpo, y_train_scenario_broad_hpo = downsample_benign_data(
            X_train_scenario_scaled_hpo,
            y_train_scenario_broad_hpo,
            broad_label_encoder,
            ratio=DOWNSAMPLE_BENIGN_RATIO_HPO
        )

        print(f"HPO subsample shape after downsampling: {X_train_scenario_scaled_hpo.shape}")

        #--- ADD THESE DEBUG PRINTS ---
        print(f"DEBUG: y_train_scenario_broad_hpo dtype: {y_train_scenario_broad_hpo.dtype}")
        print(f"DEBUG: y_train_scenario_broad_hpo unique values: {np.unique(y_train_scenario_broad_hpo).tolist()}")
        print(f"DEBUG: y_train_scenario_broad_hpo value_counts (from Series):\n{pd.Series(y_train_scenario_broad_hpo).value_counts()}")
        # Explicitly check for NaN values
        if np.isnan(y_train_scenario_broad_hpo).any():
            print("DEBUG: y_train_scenario_broad_hpo CONTAINS NaN VALUES!")
        else:
            print("DEBUG: y_train_scenario_broad_hpo DOES NOT contain NaN values.")
        # --- END DEBUG PRINTS ---


        # NEW: Calculate sample weights for imbalanced data (based on combined training data)
        print("\nCalculating sample weights for imbalanced broad labels in training data...")
        if len(np.unique(y_train_scenario_broad_hpo)) < 2:
            print(f"Skipping model training for this scenario: Only one class present in HPO subsample. Cannot calculate class weights or perform multi-class classification.")
            all_scenario_results[f"Train_{train_name}_Test_{test_name}"] = {"Error": "Single Class in HPO Subsample"}
            continue

        classes_in_hpo_train = np.unique(y_train_scenario_broad_hpo)
        class_weights_array = compute_class_weight(
            'balanced',
            classes=classes_in_hpo_train,
            y=y_train_scenario_broad_hpo
        )
        class_weights_dict = dict(zip(classes_in_hpo_train, class_weights_array))
        sample_weights_hpo = np.array([class_weights_dict[label] for label in y_train_scenario_broad_hpo])
        print("Sample weights calculated for training data.")


        scenario_results = {}

        # NEW: Create directories for saving models and results for this scenario
        scenario_key_str = f"Train_{train_name}_Test_{test_name}"
        scenario_model_dir = os.path.join(CROSS_MODELS_DIR, scenario_key_str)
        scenario_results_dir = os.path.join(CROSS_RESULTS_DIR, scenario_key_str)
        os.makedirs(scenario_model_dir, exist_ok=True)
        os.makedirs(scenario_results_dir, exist_ok=True)



        # Hyperparameter Optimization and Model Training/Evaluation
        for model_name, base_model in base_models.items():
            model_path = os.path.join(scenario_model_dir, f"{model_name}.joblib")
            model_results_path = os.path.join(scenario_results_dir, f"{model_name}_metrics.joblib")

            


            # NEW: Check if model and results are already saved
            if os.path.exists(model_path) and os.path.exists(model_results_path) and not force_retrain:
                print(f"Loading saved model and results for {model_name} from {scenario_key_str}...")
                best_model = joblib.load(model_path)
                loaded_model_results = joblib.load(model_results_path)
                scenario_results[model_name] = loaded_model_results
                scenario_results[model_name]['Trained_Model'] = best_model # Store model for per-day eval
                print("Loaded successfully.")
                continue # Skip HPO and evaluation for this model


            print(f"\n--- Hyperparameter Optimization for {model_name} (Training on {train_name}) ---")
            param_grid = PARAM_GRIDS[model_name] # Use PARAM_GRIDS from constants

            search_start_time = time.time()

            # MODIFIED: Use RandomizedSearchCV for all models
            search_class = RandomizedSearchCV
            n_iter_for_search = 20 # Default n_iter for RandomizedSearchCV (adjust as needed)

            search = search_class(
                estimator=base_model,
                param_distributions=param_grid,
                n_iter=n_iter_for_search,
                scoring=scoring_metrics,
                refit=refit_metric,
                cv=cv_strategy,
                n_jobs=-1,
                verbose=2
            )


            # MODIFIED: Pass sample_weight to GridSearchCV's fit method
            search.fit(X_train_scenario_scaled_hpo, y_train_scenario_broad_hpo, sample_weight=sample_weights_hpo)
            results_df = pd.DataFrame(search.cv_results_)
            print(results_df[['param_subsample', 'param_n_estimators', 'param_max_depth', 'param_learning_rate', 'param_colsample_bytree', 'mean_test_f1_weighted_attack_only', 'std_test_f1_weighted_attack_only']].sort_values(by='mean_test_f1_weighted_attack_only', ascending=False))



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

            # --- NEW: Check if best_model is valid immediately after HPO ---
            if best_model is None:
                print(f"ERROR: {model_name} failed to produce a best_estimator_ after HPO. Skipping further processing for this model.")
                # Store an error status for this model in the results
                scenario_results[model_name] = {"Error": "Failed to train best_estimator_"}
                continue # Skip to the next model in the base_models loop
            # --- END NEW CHECK ---



            # NEW: Apply Benign Downsampling to the FULL training data before final model fit
            # This is crucial because the best_model from HPO is refitted on the full training data.
            DOWNSAMPLE_BENIGN_RATIO_FINAL_TRAIN = 1.0 # Example: 1:1 ratio for final model training
            X_final_train_scenario, y_final_train_scenario = downsample_benign_data(
                X_train_scenario_hpo_source, # Use the full scaled training data (before HPO subsampling)
                y_train_scenario_broad, # Use the full training labels
                broad_label_encoder,
                ratio=DOWNSAMPLE_BENIGN_RATIO_FINAL_TRAIN
            )
            
            print(f"Final training data shape after downsampling: {X_final_train_scenario.shape}")
                
            # Refit the best model on the downsampled full training data
            # This ensures the model used for overall and per-day evaluation is trained on downsampled data
            final_model_sample_weights = np.array([class_weights_dict[label] for label in y_final_train_scenario])
            
            # Check if the model has a 'fit' method and refit it
            if hasattr(best_model, 'fit'):
                print(f"Refitting {model_name} on downsampled full training data...")
                best_model.fit(X_final_train_scenario, y_final_train_scenario, sample_weight=final_model_sample_weights)
                print("Refitting complete.")
            else:
                print(f"Warning: {model_name} does not have a 'fit' method. Skipping refitting on downsampled data.")

            # Store overall metrics for the generalizable model
            scenario_results[model_name] = {
                'Best Params': search.best_params_,
                'Optimization Time (s)': search_duration,
                'CV Metrics': cv_means,
                'Overall Metrics (Combined Test Set)': {}, # Populated below
                'Per_Day_Metrics': {} # Initialized for per-day results
            }
            # Store the trained model object directly in the results dictionary
            scenario_results[model_name]['Trained_Model'] = best_model
            
            # --- Overall Evaluation on Combined Test Set ---
            # This section runs for both loaded and newly trained models
            # Get the model to use for evaluation (either loaded or newly trained)
            model_for_eval = scenario_results[model_name]['Trained_Model']

            # --- Overall Evaluation on Combined Test Set ---
            print(f"\n--- Overall Evaluation of Optimized {model_name} on Combined Test Set ({test_name}) ---")
            

            # Prepare X_test_scenario_combined_df for prediction
            # Convert to NumPy array for consistent input to models
            X_test_scenario_combined_scaled = X_test_scenario_combined_df.copy()
            X_test_scenario_combined_scaled[numerical_cols_for_scaling] = scaler.transform(X_test_scenario_combined_df[numerical_cols_for_scaling])
            X_test_scenario_combined_scaled_np = X_test_scenario_combined_scaled.values
            
            
            # Make predictions and get probabilities
            y_pred_combined = best_model.predict(X_test_scenario_combined_scaled_np)
            y_proba_combined = best_model.predict_proba(X_test_scenario_combined_scaled_np)


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



            # Calculate overall metrics
            overall_accuracy = accuracy_score(y_test_scenario_combined_broad, y_pred_combined)
            overall_precision = precision_score(y_test_scenario_combined_broad, y_pred_combined, average='weighted', zero_division=0)
            overall_recall = recall_score(y_test_scenario_combined_broad, y_pred_combined, average='weighted', zero_division=0)
            overall_f1 = f1_score(y_test_scenario_combined_broad, y_pred_combined, average='weighted', zero_division=0)
            overall_balanced_accuracy = balanced_accuracy_score(y_test_scenario_combined_broad, y_pred_combined)

            # NEW: F1-Score for Attack Labels Only (Overall)
            mask_attack_labels = np.isin(y_test_scenario_combined_broad, attack_labels_encoded)
            y_true_attack_only = y_test_scenario_combined_broad[mask_attack_labels]
            y_pred_attack_only = y_pred_combined[mask_attack_labels]

            overall_f1_attack_only = np.nan
            if len(np.unique(y_true_attack_only)) >= 2:
                try:
                    overall_f1_attack_only = f1_score(
                        y_true_attack_only,
                        y_pred_attack_only,
                        labels=attack_labels_encoded,
                        average='weighted',
                        zero_division=0
                    )
                except Exception as e:
                    print(f"  Warning: Could not calculate F1-Score (Attack Only) for overall {test_name}. Error: {e}")
            else:
                print(f"  Warning: Not enough unique attack classes in overall test set for {test_name} to calculate F1-Score (Attack Only).")

            
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
            print(f"Overall F1-Score (weighted, Attack Only): {overall_f1_attack_only:.4f}") # NEW PRINT
            print(f"Overall Balanced Accuracy: {overall_balanced_accuracy:.4f}")
            print(f"Overall Specificity (avg): {overall_specificity:.4f}")
            print(f"Overall ROC AUC (weighted): {overall_roc_auc:.4f}")


            # Store all results for the current model in the current scenario
            # ✅ This preserves existing keys and only adds/updates new ones
            scenario_results[model_name].update({
                'Overall Metrics (Combined Test Set)': {
                    'Accuracy': overall_accuracy,
                    'Precision': overall_precision,
                    'Recall': overall_recall,
                    'F1-Score': overall_f1,
                    'F1-Score (Attack Only)': overall_f1_attack_only,
                    'Balanced Accuracy': overall_balanced_accuracy,
                    'Specificity': overall_specificity,
                    'ROC AUC': overall_roc_auc,
                    'Specific Attack Metrics': {}
                },
                'Per_Day_Metrics': {}
            })



            # Extract specific attack class metrics for overall test set
            
            overall_clf_report_dict = classification_report(
                y_test_scenario_combined_broad,
                y_pred_combined,
                output_dict=True,
                zero_division=0,
                labels=np.arange(len(broad_label_encoder.classes_)),  
                target_names=broad_label_encoder.classes_
            )
                
            for target_str_label in target_report_labels: # Use broad target labels
                if target_str_label in overall_clf_report_dict:
                    scenario_results[model_name]['Overall Metrics (Combined Test Set)']['Specific Attack Metrics'][target_str_label] = {
                        'Precision': overall_clf_report_dict[target_str_label]['precision'],
                        'Recall': overall_clf_report_dict[target_str_label]['recall'],
                        'F1-Score': overall_clf_report_dict[target_str_label]['f1-score'],
                        'Support': overall_clf_report_dict[target_str_label]['support']
                    }
                else:
                    scenario_results[model_name]['Overall Metrics (Combined Test Set)']['Specific Attack Metrics'][target_str_label] = {
                        'Precision': None, 'Recall': None, 'F1-Score': None, 'Support': 0
                    }
                    print(f"  Warning: Class '{target_str_label}' not found in combined test set '{test_name}'.")

            # --- Per-Day Evaluation on Test Set ---
            print(f"\n--- Per-Day Evaluation of Optimized {model_name} on Test Set ({test_name}) ---")
            individual_test_dfs = all_individual_dfs_by_dataset[test_name]

            # Retrieve the best_model for per-day evaluation (either loaded or newly trained)
            model_for_per_day_eval = model_for_eval


                
            for day_name, day_df_original in individual_test_dfs.items():
                print(f"\nEvaluating on Day: {day_name}")

                if day_df_original.empty:
                    print(f"    Skipping empty day DataFrame: {day_name}.")
                    scenario_results[model_name]['Per_Day_Metrics'][day_name] = "Skipped: Empty Day DataFrame"
                    continue

                # IMPORTANT: Create a fresh copy to avoid modifying original day_df_original
                temp_day_df = day_df_original.copy()


                # Drop Label/BroadLabel
                X_day = temp_day_df.drop(['Label', 'BroadLabel'], axis=1)
                # MODIFIED: Ensure y_day_broad is integer type
                y_day_broad = temp_day_df['BroadLabel'].astype(int)

                # Align columns to common_features and ensure all NaNs are filled with 0.0
                # Create a new DataFrame with all common_features, initialized to 0.0
                # Use np.float32 for consistency with optimized parquet types
                X_day_aligned = pd.DataFrame(0.0, index=X_day.index, columns=common_features, dtype=np.float32)

                # Copy values for columns that exist in both X_day and common_features
                cols_to_copy = X_day.columns.intersection(common_features)
                X_day_aligned[cols_to_copy] = X_day[cols_to_copy]
                
                
                # Explicitly fill any NaNs after alignment
                X_day_aligned.fillna(0.0, inplace=True)
                
                # Apply the SAME scaler (fitted on the training data)
                # Ensure the input to scaler.transform has columns in the exact order
                # of numerical_cols_for_scaling_combined
                X_day_to_scale = X_day_aligned[numerical_cols_for_scaling]

                X_day_scaled_np = scaler.transform(X_day_to_scale) # Directly transform to NumPy array
                
                # Free memory from intermediate DataFrames
                del X_day, temp_day_df, X_day_aligned, X_day_to_scale
                gc.collect()

                # Make predictions using the generalizable model
                y_pred_day = model_for_per_day_eval.predict(X_day_scaled_np)
                y_proba_day = model_for_per_day_eval.predict_proba(X_day_scaled_np)

                

                # Pad y_proba_day to match broad_label_encoder.classes_ size
                padded_y_proba_day = np.zeros((y_proba_day.shape[0], len(broad_label_encoder.classes_)))
                for i, model_predicted_int_class in enumerate(best_model.classes_):
                    if 0 <= model_predicted_int_class < len(broad_label_encoder.classes_):
                        padded_y_proba_day[:, model_predicted_int_class] = y_proba_day[:, i]
                    else:
                        print(f"Warning: Model predicted class {model_predicted_int_class} is out of range for broad_label_encoder.classes_ (0 to {len(broad_label_encoder.classes_)-1}). Skipping padding for this class.")

                # ---  DEBUG PRINTS ---
                print(f"    DEBUG (Cross-Eval): Day: {day_name}")
                print(f"    DEBUG (Cross-Eval): y_day_broad dtype: {y_day_broad.dtype}")
                print(f"    DEBUG (Cross-Eval): y_day_broad unique values: {y_day_broad.unique().tolist()}")
                print(f"    DEBUG (Cross-Eval): y_day_broad value_counts:\n{y_day_broad.value_counts()}")
                print(f"    DEBUG (Cross-Eval): len(y_day_broad.unique()): {len(y_day_broad.unique())}")
                # --- END DEBUG PRINTS ---
            
                # Calculate per-day metrics
                if len(np.unique(y_day_broad)) < 2:
                    print(f"  Warning: ROC AUC undefined for {day_name} (only one class in true labels).")
                    day_metrics = {"Error": "Single Class in Day Data"}
                else:
                    day_accuracy = accuracy_score(y_day_broad, y_pred_day)
                    day_precision = precision_score(y_day_broad, y_pred_day, average='weighted', zero_division=0)
                    day_recall = recall_score(y_day_broad, y_pred_day, average='weighted', zero_division=0)
                    day_f1 = f1_score(y_day_broad, y_pred_day, average='weighted', zero_division=0)
                    day_balanced_accuracy = balanced_accuracy_score(y_day_broad, y_pred_day)
                    
                    # NEW: F1-Score for Attack Labels Only (per day)
                    mask_attack_labels_day = np.isin(y_day_broad, attack_labels_encoded)
                    y_true_attack_only_day = y_day_broad[mask_attack_labels_day]
                    y_pred_attack_only_day = y_pred_day[mask_attack_labels_day]

                    day_f1_attack_only = np.nan
                    if len(np.unique(y_true_attack_only_day)) >= 2:
                        try:
                            day_f1_attack_only = f1_score(
                                y_true_attack_only_day,
                                y_pred_attack_only_day,
                                labels=attack_labels_encoded,
                                average='weighted',
                                zero_division=0
                            )
                        except Exception as e:
                            print(f"    Warning: Could not calculate F1-Score (Attack Only) for {day_name}. Error: {e}")
                    else:
                        print(f"    Warning: Not enough unique attack classes in day test set for {day_name} to calculate F1-Score (Attack Only).")

                    
                    day_specificity_scores = calculate_specificity(y_day_broad, y_pred_day, broad_label_encoder)
                    day_specificity = np.mean(list(day_specificity_scores.values()))

                    day_roc_auc = np.nan
                    try:
                        day_roc_auc = roc_auc_score(
                            y_day_broad,
                            padded_y_proba_day,
                            multi_class='ovr',
                            average='weighted',
                            labels=np.arange(len(broad_label_encoder.classes_))
                        )
                    except ValueError as e:
                        print(f"  Warning: Could not calculate ROC AUC for {day_name}. Error: {e}")
                    except Exception as e:
                        print(f"  Warning: Unexpected error calculating ROC AUC for {day_name}. Error: {e}")


                # Remove redundant print(classification_report(...))
                # print(f"  Accuracy: {day_accuracy:.4f}")
                # print(f"  Precision (weighted): {day_precision:.4f}")
                # print(f"  Recall (weighted): {day_recall:.4f}")
                # print(f"  F1-Score (weighted): {day_f1:.4f}")
                # print(f"  Balanced Accuracy: {day_balanced_accuracy:.4f}")
                # print(f"  Specificity (avg): {day_specificity:.4f}")
                # print(f"  ROC AUC (weighted): {day_roc_auc:.4f}")
                # print(f"\n  Per-Class Report for {day_name}:")
                
                # Extract specific attack class metrics for this day
                day_clf_report_dict = classification_report(
                    y_day_broad,
                    y_pred_day,
                    output_dict=True,
                    zero_division=0,
                    labels=np.arange(len(broad_label_encoder.classes_)),
                    target_names=broad_label_encoder.classes_
                )
                
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
                    'F1-Score (Attack Only)': day_f1_attack_only, # NEW STORE
                    'Balanced Accuracy': day_balanced_accuracy,
                    'Specificity': day_specificity,
                    'ROC AUC': day_roc_auc,
                    'Specific Attack Metrics': specific_day_metrics
                }
                del y_pred_day, y_proba_day, padded_y_proba_day # Free memory
                gc.collect() # Clean up after each day's evaluation

            print("-" * 60 + "\n")
            # NEW DEBUG PRINTS FOR SAVING ISSUE
            print(f"DEBUG SAVE: Attempting to save {model_name} for scenario {scenario_key_str}")
            print(f"DEBUG SAVE: model_path: {model_path}")
            print(f"DEBUG SAVE: model_results_path: {model_results_path}")
            
            # Check the model object itself
            model_to_save = scenario_results[model_name].get('Trained_Model')
            print(f"DEBUG SAVE: Type of model_to_save: {type(model_to_save)}")
            print(f"DEBUG SAVE: Is model_to_save None? {model_to_save is None}")

            # Check the results dictionary
            results_to_save = scenario_results[model_name]
            print(f"DEBUG SAVE: Type of results_to_save: {type(results_to_save)}")
            print(f"DEBUG SAVE: Keys in results_to_save: {list(results_to_save.keys())}")
            if 'Trained_Model' not in results_to_save:
                print("ERROR: 'Trained_Model' key is MISSING in results_to_save dictionary!")
            else:
                print("DEBUG SAVE: 'Trained_Model' key IS PRESENT in results_to_save dictionary.")

            # End of debug prints for saving issue

            # NEW: Save the model and its results *after* the per-day evaluation loop completes
            print(f"Saving model and results for {model_name} for scenario {scenario_key_str}...")
            # The error occurs here. We need the traceback.

             
            joblib.dump(scenario_results[model_name]['Trained_Model'], model_path) # Save the trained model
            joblib.dump(scenario_results[model_name], model_results_path) # Save the results dict (now complete)
            print("Saved successfully.")

        # Store results for the current scenario
        all_scenario_results[scenario_key_str] = scenario_results

    return all_scenario_results # Return the collected results


