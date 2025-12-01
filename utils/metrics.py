import json
import os
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix, fbeta_score, roc_auc_score

def evaluate_metrics(y_true, y_pred, y_probs=None, labels=None):
    """
    Calculates common classification metrics including F2 and ROC-AUC.
    
    Args:
        y_true (array-like): True labels.
        y_pred (array-like): Predicted labels.
        y_probs (array-like, optional): Predicted probabilities for ROC-AUC.
        labels (list, optional): List of class indices.
        
    Returns:
        dict: Dictionary containing accuracy, macro_f1, weighted_f1, per_class report, confusion matrix, f2_score, roc_auc.
    """
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    weighted_f1 = f1_score(y_true, y_pred, average='weighted')
    macro_f2 = fbeta_score(y_true, y_pred, beta=2, average='macro')
    
    metrics = {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "macro_f2": macro_f2,
        "classification_report": classification_report(y_true, y_pred, output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist()
    }
    
    if y_probs is not None:
        try:
            # Multi-class ROC AUC (One-vs-Rest)
            roc_auc = roc_auc_score(y_true, y_probs, multi_class='ovr', average='macro')
            metrics["roc_auc"] = roc_auc
        except Exception as e:
            print(f"Warning: Could not calculate ROC-AUC: {e}")
            metrics["roc_auc"] = None
            
    return metrics

def brier_score(y_true, y_prob, positive_class=None):
    """
    Computes Brier Score.
    If positive_class is None, computes multiclass Brier score (mean squared error of one-hot target vs probs).
    If positive_class is int, computes binary Brier score for that class vs rest.
    """
    if positive_class is not None:
        # Binary Brier for specific class
        y_true_bin = (np.array(y_true) == positive_class).astype(int)
        y_prob_bin = np.array(y_prob)[:, positive_class]
        return np.mean((y_prob_bin - y_true_bin) ** 2)
    else:
        # Multiclass Brier (sum of squared differences per sample, averaged)
        # One-hot encode y_true
        n_classes = y_prob.shape[1]
        y_true_onehot = np.eye(n_classes)[y_true]
        return np.mean(np.sum((y_prob - y_true_onehot) ** 2, axis=1))

def reliability_curve(y_true, y_prob, n_bins=10, positive_class=None):
    """
    Computes reliability curve (calibration curve) for a specific class.
    Returns: bin_centers, mean_predicted_value, fraction_of_positives
    """
    if positive_class is None:
        raise ValueError("positive_class must be specified for reliability_curve")
        
    y_true_bin = (np.array(y_true) == positive_class).astype(int)
    y_prob_bin = np.array(y_prob)[:, positive_class]
    
    from sklearn.calibration import calibration_curve
    prob_true, prob_pred = calibration_curve(y_true_bin, y_prob_bin, n_bins=n_bins, strategy='uniform')
    
    return prob_pred, prob_true

def expected_cost(y_true, y_pred, cost_matrix):
    """
    Computes Expected Cost based on a cost matrix.
    cost_matrix: dict with keys like ("fatal", "FN"), ("fatal", "FP"), ("nonfatal", "misclass")
    Assumes Class 3 is "fatal" (Severity 4).
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    total_cost = 0.0
    n = len(y_true)
    
    # Fatal Class Index
    FATAL_IDX = 3
    
    for i in range(n):
        true_label = y_true[i]
        pred_label = y_pred[i]
        
        if true_label == FATAL_IDX and pred_label != FATAL_IDX:
            # Fatal False Negative (Missed Fatal)
            total_cost += cost_matrix.get(("fatal", "FN"), 0.0)
        elif true_label != FATAL_IDX and pred_label == FATAL_IDX:
            # Fatal False Positive (False Alarm)
            total_cost += cost_matrix.get(("fatal", "FP"), 0.0)
        elif true_label != pred_label:
            # Non-fatal Misclassification (e.g. Sev 1 as Sev 2)
            total_cost += cost_matrix.get(("nonfatal", "misclass"), 0.0)
            
    return total_cost / n

def save_metrics(metrics, config, save_path):
    """
    Saves metrics and configuration to a JSON file.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    data = {
        "config": config,
        "metrics": metrics
    }
    
    with open(save_path, 'w') as f:
        json.dump(data, f, indent=4)
    
    print(f"Metrics saved to {save_path}")
