# intra_evaluation.py
import time
import pandas as pd
import numpy as np
import gc # Import for garbage collection

# NEW IMPORTS FOR SAVING/LOADING
import joblib
import os

# Scikit-learn models
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression

# Scikit-learn preprocessing and model selection
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
#from sklearn.decomposition import PCA # NEW: Import PCA
from sklearn.utils.class_weight import compute_class_weight # NEW: Import for class weights



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

# NEW CONSTANTS FOR SAVING/LOADING
MODELS_DIR = 'saved_models'
RESULTS_DIR = 'saved_results'


# NEW: Downsampling function
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
    y_series = y_series.reset_index(drop=True)

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
    # Ensure attack_df is not empty to avoid division by zero or min(0, ...)
    if len(attack_df) == 0:
        print("  No attack samples found for downsampling. Returning original data.")
        return X, y
    
    # The sample size for benign is 'ratio' times the number of attack samples
    # but capped by the actual number of benign samples available.
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


def run_intra_dataset_evaluation(
    all_combined_dfs, # Original: Combined DataFrames per year (e.g., {'CIC_IDS_2017': df})
    all_individual_dfs_by_dataset, # NEW: Individual daily DataFrames (e.g., {'CIC_IDS_2017': {'day1': df1, ...}})
    label_encoder, # Passed from main_evaluation after fitting on all labels
    ALL_ENCODED_LABELS, # All encoded granular labels
    common_features, # Passed from main_evaluation
    broad_label_mapper, # Function to map granular encoded to broad encoded
    broad_label_encoder, # LabelEncoder for broad labels
    target_report_labels, # <-- NEW PARAMETER
    force_retrain: bool = False # NEW ARGUMENT: Set to True to force retraining
):
    """
    Performs intra-dataset generalization evaluation.
    Trains a single model per dataset (e.g., per year) and evaluates it on
    individual daily test sets within that dataset.

    Args:
        all_combined_dfs (dict): Dictionary of combined DataFrames for each dataset (e.g., per year).
        all_individual_dfs_by_dataset (dict): Nested dictionary of individual daily DataFrames.
        label_encoder (LabelEncoder): Fitted LabelEncoder for consistent label transformation.
        ALL_ENCODED_LABELS (list): All encoded granular labels.
        common_features (list): List of features common across all datasets.
        broad_label_mapper (function): Function to map granular encoded to broad encoded.
        broad_label_encoder (LabelEncoder): LabelEncoder for broad labels.
        target_report_labels (list): List of broad labels for specific reporting.
        force_retrain (bool): If True, force retraining even if saved models exist.

    Returns:
        dict: A dictionary containing all evaluation results, structured per dataset and then per individual day.
    """
    print("\n--- Starting Intra-Dataset Generalization Evaluation (Single Model per Dataset, Per-Day Evaluation) ---")

    print("\n--- DEBUG (intra_evaluation): State of broad_label_encoder received ---")
    print(f"DEBUG (intra_evaluation): Type of broad_label_encoder: {type(broad_label_encoder)}")
    print("-------------------------------------------------------------------------\n")


    
    # Define base models (will be cloned by GridSearchCV)
    base_models = {
        #'Logistic Regression': LogisticRegression(random_state=42, n_jobs=-1, class_weight='balanced', solver='saga', max_iter=1000),
        # In intra_evaluation.py, base_models
        #'Random Forest': RandomForestClassifier(random_state=42, n_jobs=-1, class_weight='balanced', n_estimators=100), # Default for base model
        'XGBoost': XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='mlogloss', n_jobs=-1, tree_method='hist')
    }

    # Cross-validation strategy for HPO (k=5 stratified as per paper)
    cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Scoring metrics for GridSearchCV (as per paper)
    scoring_metrics = get_scoring_metrics(broad_label_encoder)
    refit_metric = 'f1_weighted_attack_only' # Metric to use for selecting the best model


    # This will store results structured as {dataset_name: {model_name: {overall_metrics, per_day_metrics: {day_name: metrics}}}}
    all_intra_results = {}

    # Pre-calculate integer labels for attack classes for efficiency
    # These are the labels you want to include in "attack-only" metrics
    attack_labels_string = [label for label in target_report_labels if label != 'Benign'] # Ensure Benign is not included
    attack_labels_encoded = broad_label_encoder.transform(attack_labels_string)

    # Outer loop: Iterate through each combined dataset (e.g., CIC_IDS_2017, CIC_IDS_2018)
    for dataset_name, combined_df_intra in all_combined_dfs.items():
        print(f"\n################################################################################")
        print(f"### SCENARIO: Intra-Dataset Evaluation for {dataset_name} ###")
        print(f"################################################################################")

        # ADDED: Skip if DataFrame is empty or too small for a split
        if combined_df_intra.empty:
            print(f"Skipping {dataset_name}: Combined DataFrame is empty.")
            all_intra_results[dataset_name] = "Skipped: Empty Combined DataFrame"
            continue


        X_combined = combined_df_intra.drop(['Label', 'BroadLabel'], axis=1)
        y_combined_broad = combined_df_intra['BroadLabel']


        # Impute NaNs introduced by feature alignment with 0 (before train-test split)
        # This handles columns that were missing in a dataset but present in common_features
        print("Imputing NaNs introduced by feature alignment with 0 for intra-dataset features...")
        X_combined.fillna(0, inplace=True)
        print("NaN imputation complete for intra-dataset features.")
        gc.collect()

        # Perform train-test split on the ENTIRE combined dataset for this year
        # This split defines the training data for the generalizable model and the overall test set
        if len(y_combined_broad.unique()) < 2:
            print(f"Skipping {dataset_name}: Only one class present in broad labels. Cannot perform stratified split.")
            all_intra_results[dataset_name] = "Skipped: Single Class in Combined Data"
            continue
        if len(y_combined_broad) < 2:
            print(f"Skipping {dataset_name}: Not enough samples for train-test split.")
            all_intra_results[dataset_name] = "Skipped: Not enough samples in Combined Data"
            continue




        X_train_combined, X_test_combined, y_train_combined_broad, y_test_combined_broad = train_test_split(
            X_combined, y_combined_broad, test_size=0.2, random_state=42, stratify=y_combined_broad
        )

        del X_combined, y_combined_broad # Free up memory
        gc.collect()

        
        print(f"X_train_combined shape: {X_train_combined.shape}, y_train_combined_broad shape: {y_train_combined_broad.shape}")
        print(f"X_test_combined shape: {X_test_combined.shape}, y_test_combined_broad shape: {y_test_combined_broad.shape}")

        # Z-score Normalization for this Combined Dataset Scenario
        print("\nApplying Z-score normalization for this combined dataset scenario...")
        scaler_combined = StandardScaler()
        numerical_cols_for_scaling_combined = X_train_combined.select_dtypes(include=np.number).columns.tolist()

        scaler_combined.fit(X_train_combined[numerical_cols_for_scaling_combined])

        X_train_combined_scaled = X_train_combined.copy()
        X_test_combined_scaled = X_test_combined.copy()


        X_train_combined_scaled[numerical_cols_for_scaling_combined] = scaler_combined.transform(X_train_combined[numerical_cols_for_scaling_combined])
        X_test_combined_scaled[numerical_cols_for_scaling_combined] = scaler_combined.transform(X_test_combined[numerical_cols_for_scaling_combined])
        print("Z-score normalization complete for combined dataset scenario.")

        # Convert to NumPy array for consistent input to models
        X_train_combined_scaled_np = X_train_combined_scaled.values
        X_test_combined_scaled_np = X_test_combined_scaled.values

        del X_train_combined, X_test_combined,X_train_combined_scaled, X_test_combined_scaled
        gc.collect()

        # # Apply PCA for Dimensionality Reduction (fitted on combined training data)
        # n_components_pca = 0.98
        # print(f"\nApplying PCA with n_components={n_components_pca} for combined dataset...")
        # try:
        #     pca_combined = PCA(n_components=n_components_pca, random_state=42)
        #     pca_combined.fit(X_train_combined_scaled)

        #     X_train_combined_scaled_pca = pca_combined.transform(X_train_combined_scaled)
        #     X_test_combined_scaled_pca = pca_combined.transform(X_test_combined_scaled)
        #     print(f"PCA applied. Original features: {X_train_combined_scaled.shape[1]}, PCA components: {X_train_combined_scaled_pca.shape[1]}")

        #     X_train_combined_scaled = X_train_combined_scaled_pca
        #     X_test_combined_scaled = X_test_combined_scaled_pca

        #     del X_train_combined_scaled_pca, X_test_combined_scaled_pca
        #     gc.collect()

        # except MemoryError:
        #     print("Standard PCA ran out of memory. Consider using IncrementalPCA or reducing n_components_pca.")
        #     # If standard PCA fails, you might need to implement IncrementalPCA here
        #     # For now, we'll proceed, but performance might be impacted if PCA couldn't be applied.
        #     pass
        # # END NEW PCA
        # # Create a subsample of the training data for HPO (to manage memory)
        # # Create a subsample of the combined training data for HPO
        hpo_sample_size_combined = 1000000
        
        X_train_combined_hpo_source = X_train_combined_scaled_np # Source for HPO subsample
        y_train_combined_hpo_source = y_train_combined_broad # Source for HPO subsample

        if len(X_train_combined_hpo_source) > hpo_sample_size_combined:
            print(f"\nSubsampling combined training data for HPO to {hpo_sample_size_combined} rows...")
            X_train_combined_scaled_hpo, _, y_train_combined_broad_hpo , _ = train_test_split(
                X_train_combined_hpo_source, y_train_combined_hpo_source,
                train_size=hpo_sample_size_combined, stratify=y_train_combined_hpo_source, random_state=42
            )
            print(f"HPO subsample shape: {X_train_combined_scaled_hpo.shape}")
            print(f"DEBUG: y_train_combined_broad_hpo distribution *AFTER* train_test_split:")
            print(pd.Series(y_train_combined_broad_hpo).value_counts())
            print(f"DEBUG: y_train_combined_broad_hpo unique values *AFTER* train_test_split: {np.unique(y_train_combined_broad_hpo).tolist()}")
            print(f"DEBUG: y_train_combined_broad_hpo length *AFTER* train_test_split: {len(y_train_combined_broad_hpo)}")
            
        else:
            X_train_combined_scaled_hpo = X_train_combined_hpo_source
            y_train_combined_broad_hpo = y_train_combined_hpo_source
            print("Combined training data size is small enough, no subsampling for HPO.")


        # NEW: Apply Benign Downsampling to the HPO subsample
        DOWNSAMPLE_BENIGN_RATIO_HPO = 1.0 # Example: 1:1 ratio of attack to benign for HPO
        X_train_combined_scaled_hpo, y_train_combined_broad_hpo = downsample_benign_data(
            X_train_combined_scaled_hpo,
            y_train_combined_broad_hpo,
            broad_label_encoder,
            ratio=DOWNSAMPLE_BENIGN_RATIO_HPO
        )

        print(f"HPO subsample shape after downsampling: {X_train_combined_scaled_hpo.shape}")
        print(f"HPO subsample label distribution after downsampling:\n{pd.Series(y_train_combined_broad_hpo).value_counts()}")


        # Calculate sample weights for imbalanced data (based on combined training data)
        print("\nCalculating sample weights for imbalanced broad labels in combined training data...")
        if len(np.unique(y_train_combined_broad_hpo)) < 2:
            print(f"Skipping model training for {dataset_name}: Only one class present in combined HPO subsample. Cannot calculate class weights or perform multi-class classification.")
            all_intra_results[dataset_name] = {"Error": "Single Class in Combined HPO Subsample"}
            continue


        classes_in_hpo_train = np.unique(y_train_combined_broad_hpo)
        class_weights_array = compute_class_weight(
            'balanced',
            classes=classes_in_hpo_train,
            y=y_train_combined_broad_hpo
        )
        class_weights_dict = dict(zip(classes_in_hpo_train, class_weights_array))
        sample_weights_hpo = np.array([class_weights_dict[label] for label in y_train_combined_broad_hpo])
        print("Sample weights calculated for combined training data.")

        # Initialize storage for this dataset's results
        current_dataset_results = {}

        # NEW: Create directories for saving models and results for this dataset
        dataset_model_dir = os.path.join(MODELS_DIR, dataset_name)
        dataset_results_dir = os.path.join(RESULTS_DIR, dataset_name)
        os.makedirs(dataset_model_dir, exist_ok=True)
        os.makedirs(dataset_results_dir, exist_ok=True)


        # Hyperparameter Optimization and Model Training (for generalizable model)
        for model_name, base_model in base_models.items():
            model_path = os.path.join(dataset_model_dir, f"{model_name}.joblib")
            model_results_path = os.path.join(dataset_results_dir, f"{model_name}_metrics.joblib")

            # NEW: Check if model and results are already saved
            if os.path.exists(model_path) and os.path.exists(model_results_path) and not force_retrain:
                print(f"Loading saved model and results for {model_name} from {dataset_name}...")
                best_model = joblib.load(model_path)
                loaded_model_results = joblib.load(model_results_path)
                current_dataset_results[model_name] = loaded_model_results
                # Ensure the loaded model is stored for subsequent per-day evaluation
                # This is crucial because the per-day loop happens *after* this block
                current_dataset_results[model_name]['Trained_Model'] = best_model
                print("Loaded successfully.")
                # --- CRITICAL CHANGE: Add 'continue' here ---
                # If loaded, skip the rest of this model's processing and move to the next model
                continue 

            # If not loaded (or force_retrain is True), proceed with HPO and training
             
            print(f"\n--- Hyperparameter Optimization for {model_name} (Combined Dataset: {dataset_name}) ---")
            param_grid = PARAM_GRIDS[model_name]

        

            search_start_time = time.time()

            
            search_class = RandomizedSearchCV
            # MODIFIED: Reduce n_iter for Random Forest and XGBoost
            if model_name == 'Random Forest':
                n_iter_for_search = 30 # Try 30 iterations for RF  
            elif model_name == 'XGBoost':
                n_iter_for_search = 10 # Try 10 iterations for XGBoost (was 50)
            # For Logistic Regression, n_iter=20 was already set.
            else:
                n_iter_for_search = 20 # Default for other models if added later

            
            # If you specifically want GridSearchCV for a model, you can add an else if:
            # elif model_name == 'SomeOtherModel':
            #     search_class = GridSearchCV

            search = search_class( # Use the chosen search class (RandomizedSearchCV)
                estimator=base_model,
                param_distributions=param_grid, # Use param_distributions for RandomizedSearchCV
                n_iter=n_iter_for_search, # Use the defined n_iter
                scoring=scoring_metrics,
                refit=refit_metric,
                cv=cv_strategy,
                n_jobs=-1,
                verbose=2,
                random_state=42 # For reproducibility
            )
            
            
            # Fit GridSearchCV on the HPO subsample
            # MODIFIED: Pass sample_weight to GridSearchCV's fit method
            search.fit(X_train_combined_scaled_hpo, y_train_combined_broad_hpo, sample_weight=sample_weights_hpo)

            search_end_time = time.time()
            search_duration = search_end_time - search_start_time

            print(f"Hyperparameter optimization for {model_name} completed in {search_duration:.2f} seconds.")
            print(f"Best parameters for {model_name}: {search.best_params_}")
            print(f"Best cross-validation score ({refit_metric}) for {model_name}: {search.best_score_:.4f}")

            # Store mean CV metrics as per paper
            cv_means = {}
            for metric_key in scoring_metrics.keys():
                cv_result_key = f'mean_test_{metric_key}'
                if cv_result_key in search.cv_results_:
                    cv_means[metric_key] = search.cv_results_[cv_result_key][search.best_index_]
                    print(f"  Mean CV {metric_key}: {cv_means[metric_key]:.4f}")
                else:
                    cv_means[metric_key] = np.nan
                    print(f"  Mean CV {metric_key}: N/A (Scoring failed or key not found)")


            # Get the best model found by GridSearchCV
            best_model = search.best_estimator_

            # NEW: Apply Benign Downsampling to the FULL training data before final model fit
            # This is crucial because the best_model from HPO is refitted on the full training data.
            DOWNSAMPLE_BENIGN_RATIO_FINAL_TRAIN = 1.0 # Example: 1:1 ratio for final model training
            X_final_train, y_final_train = downsample_benign_data(
                X_train_combined_hpo_source, # Use the full scaled training data (before HPO subsampling)
                y_train_combined_broad, # Use the full training labels
                broad_label_encoder,
                ratio=DOWNSAMPLE_BENIGN_RATIO_FINAL_TRAIN
            )
            print(f"Final training data shape after downsampling: {X_final_train.shape}")

            # Refit the best model on the downsampled full training data
            # This ensures the model used for overall and per-day evaluation is trained on downsampled data
            final_model_sample_weights = np.array([class_weights_dict[label] for label in y_final_train]) # Use weights from HPO
            
            # Check if the model has a 'fit' method and refit it
            if hasattr(best_model, 'fit'):
                print(f"Refitting {model_name} on downsampled full training data...")
                best_model.fit(X_final_train, y_final_train, sample_weight=final_model_sample_weights)
                print("Refitting complete.")
            else:
                print(f"Warning: {model_name} does not have a 'fit' method. Skipping refitting on downsampled data.")

            # --- Overall Evaluation on Combined Test Set ---
            # This section runs for both loaded and newly trained models
            # Get the model to use for evaluation (either loaded or newly trained)
            model_for_eval = best_model # Use the best_model that was just trained/refitted

    
            # --- Evaluation of Generalizable Model on Overall Test Set ---
            print(f"\n--- Overall Evaluation of Optimized {model_name} on Overall Test Set ({dataset_name}) ---")
            y_pred_overall = best_model.predict(X_test_combined_scaled_np)
            y_proba_overall = best_model.predict_proba(X_test_combined_scaled_np)

            padded_y_proba_overall = np.zeros((y_proba_overall.shape[0], len(broad_label_encoder.classes_)))
            for i, model_predicted_int_class in enumerate(best_model.classes_):
                if 0 <= model_predicted_int_class < len(broad_label_encoder.classes_):
                    padded_y_proba_overall[:, model_predicted_int_class] = y_proba_overall[:, i]
                else:
                    print(f"Warning: Model predicted class {model_predicted_int_class} is out of range for broad_label_encoder.classes_ (0 to {len(broad_label_encoder.classes_)-1}). Skipping padding for this class.")
            

            intra_accuracy_overall = accuracy_score(y_test_combined_broad, y_pred_overall)
            intra_precision_overall = precision_score(y_test_combined_broad, y_pred_overall, average='weighted', zero_division=0)
            intra_recall_overall = recall_score(y_test_combined_broad, y_pred_overall, average='weighted', zero_division=0)
            intra_f1_overall = f1_score(y_test_combined_broad, y_pred_overall, average='weighted', zero_division=0)
            intra_balanced_accuracy_overall = balanced_accuracy_score(y_test_combined_broad, y_pred_overall)

            # F1-Score for Attack Labels Only (Overall)
            mask_attack_labels = np.isin(y_test_combined_broad, attack_labels_encoded)
            y_true_attack_only = y_test_combined_broad[mask_attack_labels]
            y_pred_attack_only = y_pred_overall[mask_attack_labels]

            intra_f1_attack_only = np.nan
            if len(np.unique(y_true_attack_only)) >= 2: # Need at least 2 classes for meaningful F1
                try:
                    intra_f1_attack_only = f1_score(
                        y_true_attack_only,
                        y_pred_attack_only,
                        labels=attack_labels_encoded, # Explicitly use only attack labels
                        average='weighted',
                        zero_division=0
                    )
                except Exception as e:
                    print(f"  Warning: Could not calculate F1-Score (Attack Only) for {dataset_name}. Error: {e}")
            else:
                print(f"  Warning: Not enough unique attack classes in overall test set for {dataset_name} to calculate F1-Score (Attack Only).")



            intra_specificity_scores_overall = calculate_specificity(y_test_combined_broad, y_pred_overall, broad_label_encoder)
            intra_specificity_overall = np.mean(list(intra_specificity_scores_overall.values()))




            # --- MODIFIED: Use padded_y_proba_intra for ROC AUC calculation ---
            intra_roc_auc = np.nan # Default to NaN if calculation fails
            try:
                if len(np.unique(y_test_combined_broad)) > 1:
                    intra_roc_auc_overall = roc_auc_score(
                        y_test_combined_broad,
                        padded_y_proba_overall,
                        multi_class='ovr',
                        average='weighted',
                        labels=np.arange(len(broad_label_encoder.classes_))
                    )
                else:
                    print(f"  Warning: Overall ROC AUC undefined (only one class in true labels for {dataset_name}).")
            except ValueError as e:
                print(f"  Warning: Could not calculate overall ROC AUC for {dataset_name}. Error: {e}")
            except Exception as e:
                print(f"  Warning: Unexpected error calculating overall ROC AUC for {dataset_name}. Error: {e}")
            
            print(f"Overall Accuracy: {intra_accuracy_overall:.4f}")
            print(f"Overall Precision (weighted): {intra_precision_overall:.4f}")
            print(f"Overall Recall (weighted): {intra_recall_overall:.4f}")
            print(f"Overall F1-Score (weighted): {intra_f1_overall:.4f}")
            print(f"Overall F1-Score (weighted, Attack Only): {intra_f1_attack_only:.4f}") # NEW PRINT
            print(f"Overall Balanced Accuracy: {intra_balanced_accuracy_overall:.4f}")
            print(f"Overall Specificity (avg): {intra_specificity_overall:.4f}")
            print(f"Overall ROC AUC (weighted): {intra_roc_auc_overall:.4f}")

            # Store overall metrics for the generalizable model
            current_dataset_results[model_name] = {
                'Best Params': search.best_params_,
                'Optimization Time (s)': search_duration,
                'CV Metrics': cv_means,
                'Overall Test Metrics': { # New key for overall metrics
                    'Accuracy': intra_accuracy_overall,
                    'Precision': intra_precision_overall,
                    'Recall': intra_recall_overall,
                    'F1-Score': intra_f1_overall,
                    'F1-Score (Attack Only)': intra_f1_attack_only, # NEW STORE
                    'Balanced Accuracy': intra_balanced_accuracy_overall,
                    'Specificity': intra_specificity_overall,
                    'ROC AUC': intra_roc_auc_overall,
                },
                'Per_Day_Test_Metrics': {} # Initialize for per-day results
            }

            # NEW: Store the trained model object directly in the results dictionary
            current_dataset_results[model_name]['Trained_Model'] = best_model

            


            # --- Per-Day Evaluation Loop ---
            # This loop must run regardless of whether the model was trained or loaded
            # The 'model_for_per_day_eval' will correctly point to the trained or loaded model
            print(f"\n--- Evaluating Generalizable Model on Individual Daily Test Sets for {dataset_name} ---")
            individual_dfs_for_this_dataset = all_individual_dfs_by_dataset.get(dataset_name, {})

            if not individual_dfs_for_this_dataset:
                print(f"No individual daily DataFrames found for {dataset_name}. Skipping per-day evaluation.")
                current_dataset_results[model_name]['Per_Day_Test_Metrics']['No_Days_Found'] = "Skipped: No individual daily DataFrames"
                # IMPORTANT: If no individual days, we still need to save the overall results
                # This is where the save should happen for this model
                joblib.dump(current_dataset_results[model_name]['Trained_Model'], model_path) # Save the trained model
                joblib.dump(current_dataset_results[model_name], model_results_path) # Save the results dict (now complete)
                continue # Move to next model if no days to evaluate
                

            # Retrieve the best_model for per-day evaluation (either loaded or newly trained)
            # This ensures the correct model instance is used
            model_for_per_day_eval = current_dataset_results[model_name]['Trained_Model']

            for day_name, day_df_original in individual_dfs_for_this_dataset.items():
                print(f"\n  Evaluating on Day: {day_name}")

                if day_df_original.empty:
                    print(f"    Skipping empty day DataFrame: {day_name}.")
                    current_dataset_results[model_name]['Per_Day_Test_Metrics'][day_name] = "Skipped: Empty Day DataFrame"
                    continue
                # IMPORTANT: Create a fresh copy to avoid modifying original day_df_original
                temp_day_df = day_df_original.copy()

                # Prepare day_df for prediction using the SAME scaler and PCA fitted on combined training data
                X_day = temp_day_df.drop(['Label', 'BroadLabel'], axis=1)
                y_day_broad = temp_day_df['BroadLabel'].astype(int)

                # --- ADD THESE TWO LINES FOR DEBUGGING ---
                print(f"    DEBUG: y_day_broad value_counts for {day_name}:\n{y_day_broad.value_counts()}")
                print(f"    DEBUG: y_day_broad unique values for {day_name}: {y_day_broad.unique().tolist()}")
                # --- END DEBUGGING LINES ---

                # Align columns to common_features and ensure all NaNs are filled with 0.0
                # Use X_day.dtypes[0] to infer a dtype from the DataFrame, assuming consistency
                # Or use np.float32 if you want to force it, but user says parquet is already optimized.
                X_day_aligned = pd.DataFrame(0.0, index=X_day.index, columns=common_features, dtype=X_day.dtypes[0])
                

                # Copy values for columns that exist in both X_day and common_features
                cols_to_copy = X_day.columns.intersection(common_features)
                X_day_aligned[cols_to_copy] = X_day[cols_to_copy]

                # Ensure any remaining NaNs (e.g., from original data if any, though fillna(0) should handle) are gone
                X_day_aligned.fillna(0.0, inplace=True) # Explicitly fill any NaNs after alignment

                # Apply the SAME scaler (fitted on the combined training data)
                # This is the crucial fix for the ValueError: feature names/order mismatch
                # Ensure the input to scaler.transform has columns in the exact order
                # of numerical_cols_for_scaling_combined
                X_day_to_scale = X_day_aligned[numerical_cols_for_scaling_combined]

                X_day_scaled_np = scaler_combined.transform(X_day_to_scale) # Directly transform to NumPy array
                
                # Free memory from intermediate DataFrames
                del X_day, temp_day_df, X_day_aligned,X_day_to_scale
                gc.collect()

                

    
                # Make predictions using the generalizable model
                # X_day_scaled is now the input, not X_day_scaled_pca
                y_pred_day = model_for_per_day_eval.predict(X_day_scaled_np) # Use model_for_per_day_eval
                y_proba_day = model_for_per_day_eval.predict_proba(X_day_scaled_np) # Use model_for_per_day_eval
                del X_day_scaled_np # Free memory
                gc.collect()

                # Pad probabilities (same logic as before)
                padded_y_proba_day = np.zeros((y_proba_day.shape[0], len(broad_label_encoder.classes_)))
                for i, model_predicted_int_class in enumerate(model_for_per_day_eval.classes_):
                    if 0 <= model_predicted_int_class < len(broad_label_encoder.classes_):
                        padded_y_proba_day[:, model_predicted_int_class] = y_proba_day[:, i]
                    else:
                        print(f"    Warning: Model predicted class {model_predicted_int_class} is out of range for broad_label_encoder.classes_ (0 to {len(broad_label_encoder.classes_)-1}). Skipping padding for this class.")

                # Calculate per-day metrics
                if len(y_day_broad.unique()) < 2:
                    print(f"    Skipping detailed metrics for {day_name}: Only one class present in broad labels.")
                    day_metrics = {"Error": "Single Class in Day Data"}
                else:
                    day_accuracy = accuracy_score(y_day_broad, y_pred_day)
                    day_precision = precision_score(y_day_broad, y_pred_day, average='weighted', zero_division=0)
                    day_recall = recall_score(y_day_broad, y_pred_day, average='weighted', zero_division=0)
                    day_f1 = f1_score(y_day_broad, y_pred_day, average='weighted', zero_division=0)
                    day_balanced_accuracy = balanced_accuracy_score(y_day_broad, y_pred_day)


                    # F1-Score for Attack Labels Only (per day)
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
                        print(f"    Warning: Could not calculate ROC AUC for {day_name}. Error: {e}")
                    except Exception as e:
                        print(f"    Warning: Unexpected error calculating ROC AUC for {day_name}. Error: {e}")

                    

                    # Per-class report for the day
                    day_clf_report_dict = classification_report(
                        y_day_broad,
                        y_pred_day,
                        output_dict=True,
                        zero_division=0,
                        labels=np.arange(len(broad_label_encoder.classes_)),
                        target_names=broad_label_encoder.classes_
                    )
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
                            print(f"    Warning: Class '{target_str_label}' not found in {day_name}.")

                    day_metrics = {
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
                
                # Store per-day metrics
                current_dataset_results[model_name]['Per_Day_Test_Metrics'][day_name] = day_metrics
                del y_day_broad, y_pred_day, y_proba_day, padded_y_proba_day # Free memory
                gc.collect()

            # NEW: Save the model and its results *after* the per-day evaluation loop completes
            print(f"Saving model and results for {model_name} to {dataset_name}...")
            joblib.dump(current_dataset_results[model_name]['Trained_Model'], model_path) # Save the trained model
            joblib.dump(current_dataset_results[model_name], model_results_path) # Save the results dict (now complete)
            print("Saved successfully.")
        

        all_intra_results[dataset_name] = current_dataset_results
    return all_intra_results
