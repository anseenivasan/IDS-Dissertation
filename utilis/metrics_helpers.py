# utilis/metrics_helpers.py
import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score, precision_score, recall_score, f1_score, balanced_accuracy_score
from sklearn.metrics import make_scorer # Ensure make_scorer is imported here
import warnings
from sklearn.exceptions import UndefinedMetricWarning

def calculate_specificity(y_true, y_pred, broad_label_encoder):
    """
    Calculates specificity for each class given true labels, predicted labels,
    and the broad label encoder object.
    
    Args:
        y_true (array-like): True labels (integer encoded).
        y_pred (array-like): Predicted labels (integer encoded).
        broad_label_encoder (LabelEncoder): The fitted LabelEncoder object for broad labels.
                                            Used to get all possible integer labels and their string names.
    Returns:
        dict: A dictionary where keys are string class names and values are their specificity scores.
    """
    # --- DEBUG PRINTS START ---
    # print(f"\n--- DEBUG: calculate_specificity called ---")
    # print(f"DEBUG: Type of broad_label_encoder: {type(broad_label_encoder)}")
    # if hasattr(broad_label_encoder, 'classes_'):
    #     print(f"DEBUG: broad_label_encoder.classes_ (all): {broad_label_encoder.classes_.tolist()}")
    #     print(f"DEBUG: broad_label_encoder.classes_ (length): {len(broad_label_encoder.classes_)}")
    # else:
    #     print(f"DEBUG: broad_label_encoder has no 'classes_' attribute.")
    #     return {} # Or raise an error, depending on desired robustness

    # print(f"DEBUG: y_true unique values: {np.unique(y_true).tolist()}")
    # print(f"DEBUG: y_pred unique values: {np.unique(y_pred).tolist()}")
    # --- DEBUG PRINTS END ---

    all_possible_string_labels = broad_label_encoder.classes_
    all_possible_integer_labels = np.arange(len(all_possible_string_labels))
    if len(y_true) == 0 or len(np.intersect1d(y_true, all_possible_integer_labels)) == 0:
        # Return NaN for all classes if no valid true labels are present
        return {str_label: np.nan for str_label in all_possible_string_labels}
    # Build confusion matrix using the full set of possible integer labels.
    # If a label is not present in y_true/y_pred, its row/col in CM will be zeros.
    cm = confusion_matrix(y_true, y_pred, labels=all_possible_integer_labels)
    specificity_scores = {}
    for i, label_int in enumerate(all_possible_integer_labels):
        # Get the string representation for the current integer label for the output dictionary
        label_str = broad_label_encoder.inverse_transform([label_int])[0]
        TP = cm[i, i]
        FN = np.sum(cm[i, :]) - TP
        FP = np.sum(cm[:, i]) - TP
        # Total negatives for class 'label_int' are all samples not of class 'label_int'.
        
        total_samples = np.sum(cm)
        TN = total_samples - (TP + FN + FP)
        
        specificity = TN / (TN + FP) if (TN + FP) != 0 else 0.0
        specificity_scores[label_str] = specificity # Store by string label
    return specificity_scores


# Custom ROC AUC scorer wrapper
# MODIFIED: Added 'labels_encoder' and 'sample_weight=None' to function signature
def _roc_auc_scorer_wrapper(y_true, y_score, labels_encoder, sample_weight=None, **kwargs):
    """
    Wrapper for roc_auc_score to be used with make_scorer.
    y_score will be probabilities because needs_proba=True is set in make_scorer.
    The 'labels_encoder' argument is explicitly passed to derive all possible integer labels.
    'sample_weight' is passed to roc_auc_score.
    **kwargs captures any other arguments passed by make_scorer and filters out 'needs_proba'.
    """
    # Debugging prints (can be removed after confirming fix)
    # print(f"DEBUG ROC AUC Wrapper: y_true unique: {np.unique(y_true).tolist()}")
    # print(f"DEBUG ROC AUC Wrapper: y_score shape: {y_score.shape}")
    # print(f"DEBUG ROC AUC Wrapper: labels_encoder classes: {labels_encoder.classes_.tolist()}")
    # print(f"DEBUG ROC AUC Wrapper: sample_weight provided: {sample_weight is not None}")

    # Handle single-class folds (ROC AUC undefined)
    if len(np.unique(y_true)) < 2:
        # print("DEBUG ROC AUC Wrapper: y_true has less than 2 unique classes, returning NaN.")
        return np.nan 

    # Robustly handle 1D y_score for binary classification cases (if multi_class='ovr' expects 2D)
    if y_score.ndim == 1:
        if len(np.unique(y_true)) == 2: # Confirm it's a binary classification problem in this fold
            y_score_2d = np.vstack([1 - y_score, y_score]).T
            # print(f"DEBUG ROC AUC Wrapper: Reshaped 1D y_score to 2D for binary case. New shape: {y_score_2d.shape}")
            y_score = y_score_2d
        else:
            # This case means y_score is 1D but y_true has more than 2 classes, which is unexpected.
            # print(f"DEBUG ROC AUC Wrapper: y_score is 1D but y_true has >2 classes. This is unexpected. Returning NaN.")
            return np.nan # ROC AUC is undefined or input is malformed

    # The labels argument for roc_auc_score should be the full set of possible integer labels
    # that the encoder knows about (0 to n_classes-1).
    roc_auc_labels = np.arange(len(labels_encoder.classes_))

    # Filter out 'needs_proba' from kwargs as roc_auc_score does not accept it
    filtered_kwargs = {k: v for k, v in kwargs.items() if k != 'needs_proba'}

    try:
        # MODIFIED: Pass sample_weight to roc_auc_score
        result = roc_auc_score(
            y_true,
            y_score,
            average='weighted',
            multi_class='ovr',
            labels=roc_auc_labels, # Use the derived labels
            sample_weight=sample_weight, # <--- CRITICAL CHANGE: Pass sample_weight
            **filtered_kwargs
        )
        # print(f"DEBUG ROC AUC Wrapper: Successfully calculated AUC.")
        return result
    except ValueError as e:
        # print(f"DEBUG ROC AUC Wrapper: ValueError during roc_auc_score: {e}")
        return np.nan
    except Exception as e: # Catch any other unexpected errors
        # print(f"DEBUG ROC AUC Wrapper: Caught unexpected error during roc_auc_score: {type(e).__name__}: {e}")
        # print(f"DEBUG ROC AUC Wrapper: y_true unique (at error): {np.unique(y_true).tolist()}")
        # print(f"DEBUG ROC AUC Wrapper: y_score shape (at error): {y_score.shape}")
        return np.nan


def _f1_attack_only_scorer_wrapper(y_true, y_pred, labels_encoder, average='weighted', sample_weight=None, **kwargs):
    """
    Wrapper for f1_score that calculates F1 only for attack classes.
    Assumes 'Benign' is the class to exclude.
    MODIFIED: Now accepts sample_weight and passes it to f1_score.
    """
    benign_id = labels_encoder.transform(['Benign'])[0]
    all_attack_labels_encoded = [lbl for lbl in np.arange(len(labels_encoder.classes_)) if lbl != benign_id]

    # Filter y_true and y_pred to include only attack labels
    mask_attack_labels = np.isin(y_true, all_attack_labels_encoded)
    y_true_filtered = y_true[mask_attack_labels]
    y_pred_filtered = y_pred[mask_attack_labels]

    # MODIFIED: Filter sample_weight as well
    sample_weight_filtered = sample_weight[mask_attack_labels] if sample_weight is not None else None


    if len(y_true_filtered) == 0:
        return 0.0
    
    # If only one unique attack class is present, F1-score weighted/macro is not well-defined
    # for averaging across multiple classes. Return 0.0 for HPO purposes.
    if len(np.unique(y_true_filtered)) < 2: # <--- MODIFIED CONDITION
        return 0.0 # <--- CRITICAL CHANGE: Return 0.0 instead of np.nan
    
    # NEW ADDITION START: Filter out 'needs_proba' from kwargs as f1_score does not accept it
    filtered_kwargs = {k: v for k, v in kwargs.items() if k != 'needs_proba'}
    # NEW ADDITION END

    try:
        return f1_score(
            y_true_filtered,
            y_pred_filtered,
            labels=all_attack_labels_encoded,
            average=average,
            zero_division=0,
            sample_weight=sample_weight_filtered,
            **filtered_kwargs # <--- CRITICAL CHANGE: Pass filtered_kwargs here
        )
    except Exception as e:
        print(f"DEBUG: Error in _f1_attack_only_scorer_wrapper: {e}") # Keep this for now
        return np.nan


# Factory function for scoring metrics
# MODIFIED: Changed argument name from 'all_encoded_labels' to 'broad_label_encoder'
def get_scoring_metrics(broad_label_encoder):
    """
    Returns a dictionary of scoring metrics, including a dynamically configured
    ROC AUC scorer and an F1 scorer that excludes benign.
    """
    scorers = {
        'balanced_accuracy': make_scorer(balanced_accuracy_score),
        'precision_weighted': make_scorer(precision_score, average='weighted', zero_division=0),
        'recall_weighted': make_scorer(recall_score, average='weighted', zero_division=0),
        'f1_weighted': make_scorer(f1_score, average='weighted', zero_division=0), # Keep this for overall view
        'roc_auc_weighted': make_scorer(
            _roc_auc_scorer_wrapper,
            response_method='predict_proba',
            needs_proba=True,
            labels_encoder=broad_label_encoder
        ),
        'f1_weighted_attack_only': make_scorer(
            _f1_attack_only_scorer_wrapper,
            labels_encoder=broad_label_encoder,
            average='weighted', # Default to weighted, can be changed to 'macro'
            needs_proba=False # F1 score needs predictions, not probabilities
        )
    }

    return scorers

