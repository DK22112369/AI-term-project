import json
import os
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

def evaluate_metrics(y_true, y_pred, labels=None):
    """
    Calculates common classification metrics.
    
    Args:
        y_true (array-like): True labels.
        y_pred (array-like): Predicted labels.
        labels (list, optional): List of class indices.
        
    Returns:
        dict: Dictionary containing accuracy, macro_f1, weighted_f1, per_class report, and confusion matrix.
    """
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    weighted_f1 = f1_score(y_true, y_pred, average='weighted')
    
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    
    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "classification_report": report,
        "confusion_matrix": cm.tolist() # Convert to list for JSON serialization
    }

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
