# utils/metrics_helpers.py
import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score, precision_score, recall_score, f1_score, balanced_accuracy_score
from sklearn.metrics import make_scorer # Ensure make_scorer is imported here

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
    print(f"\n--- DEBUG: calculate_specificity called ---")
    print(f"DEBUG: Type of broad_label_encoder: {type(broad_label_encoder)}")
    if hasattr(broad_label_encoder, 'classes_'):
        print(f"DEBUG: broad_label_encoder.classes_ (all): {broad_label_encoder.classes_.tolist()}")
        print(f"DEBUG: broad_label_encoder.classes_ (length): {len(broad_label_encoder.classes_)}")
    else:
        print(f"DEBUG: broad_label_encoder has no 'classes_' attribute.")
        # This should not happen if it's a LabelEncoder object.
        return {} # Or raise an error, depending on desired robustness

    print(f"DEBUG: y_true unique values: {np.unique(y_true).tolist()}")
    print(f"DEBUG: y_pred unique values: {np.unique(y_pred).tolist()}")
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
# It needs to accept **kwargs to catch any extra arguments make_scorer might pass, like 'needs_proba'
def _roc_auc_scorer_wrapper(y_true, y_score, labels, **kwargs):
    """
    Wrapper for roc_auc_score to be used with make_scorer.
    y_score will be probabilities because needs_proba=True is set in make_scorer.
    The 'labels' argument is explicitly passed to roc_auc_score to handle all classes.
    **kwargs captures any other arguments passed by make_scorer (like needs_proba)
    and filters out 'needs_proba' before passing to roc_auc_score.
    """
    # Debugging prints (can be removed after confirming fix)
    # print(f"DEBUG ROC AUC Wrapper: y_true unique: {np.unique(y_true).tolist()}")
    # print(f"DEBUG ROC AUC Wrapper: y_score shape: {y_score.shape}")
    # print(f"DEBUG ROC AUC Wrapper: labels: {labels.tolist() if isinstance(labels, np.ndarray) else labels}")

    # Handle single-class folds (ROC AUC undefined)
    if len(np.unique(y_true)) < 2:
        print("DEBUG ROC AUC Wrapper: y_true has less than 2 unique classes, returning NaN.")
        return np.nan 

    # --- NEW: Robustly handle 1D y_score for binary classification cases ---
    # This is the core fix for the AxisError
    if y_score.ndim == 1:
        # If y_score is 1D, it's likely the probability of the positive class in a binary scenario.
        # We need to convert it to (n_samples, 2) for multi_class='ovr'
        if len(np.unique(y_true)) == 2: # Confirm it's a binary classification problem in this fold
            #Create a 2D array: column 0 is prob of negative class, column 1 is prob of positive class
            y_score_2d = np.vstack([1 - y_score, y_score]).T
            print(f"DEBUG ROC AUC Wrapper: Reshaped 1D y_score to 2D for binary case. New shape: {y_score_2d.shape}")
            y_score = y_score_2d
        else:
            # This case means y_score is 1D but y_true has more than 2 classes, which is unexpected.
            # print(f"DEBUG ROC AUC Wrapper: y_score is 1D but y_true has >2 classes. This is unexpected. Returning NaN.")
            return np.nan # ROC AUC is undefined or input is malformed

    # Filter out 'needs_proba' from kwargs as roc_auc_score does not accept it
    filtered_kwargs = {k: v for k, v in kwargs.items() if k != 'needs_proba'}

    try:
        result = roc_auc_score(y_true, y_score, average='weighted', multi_class='ovr', labels=labels, **filtered_kwargs)
        # print(f"DEBUG ROC AUC Wrapper: Successfully calculated AUC.")
        return result
    except ValueError as e:
        # This catches cases where roc_auc_score itself raises ValueError (e.g., if y_true/y_score are malformed)
        # print(f"DEBUG ROC AUC Wrapper: ValueError during roc_auc_score: {e}")
        return np.nan
    except Exception as e: # Catch any other unexpected errors, including the AxisError
        # print(f"DEBUG ROC AUC Wrapper: Caught unexpected error during roc_auc_score: {type(e).__name__}: {e}")
        # print(f"DEBUG ROC AUC Wrapper: y_true unique (at error): {np.unique(y_true).tolist()}")
        # print(f"DEBUG ROC AUC Wrapper: y_score shape (at error): {y_score.shape}")
        return np.nan # Return NaN to prevent GridSearchCV from crashing.


# Factory function for scoring metrics
def get_scoring_metrics(all_encoded_labels):
    """
    Returns a dictionary of scoring metrics, including a dynamically configured
    ROC AUC scorer that uses the provided all_encoded_labels.
    """
    return {
        'balanced_accuracy': make_scorer(balanced_accuracy_score),
        'precision_weighted': make_scorer(precision_score, average='weighted', zero_division=0),
        'recall_weighted': make_scorer(recall_score, average='weighted', zero_division=0),
        'f1_weighted': make_scorer(f1_score, average='weighted', zero_division=0),
        'roc_auc_weighted': make_scorer(_roc_auc_scorer_wrapper, needs_proba=True, labels=all_encoded_labels)
    }
