"""
Classification metrics utilities.

Computes evaluation metrics for classification:
- Confusion Matrix (TP, TN, FP, FN)
- Precision
- Recall
- F-measure (F1 score)
"""

import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)
from typing import Union


def compute_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray,
                             labels: list = None) -> dict:
    """
    Compute confusion matrix and extract TP, TN, FP, FN.
    
    For binary classification:
    - TP (True Positives): Correctly predicted positive
    - TN (True Negatives): Correctly predicted negative
    - FP (False Positives): Incorrectly predicted positive
    - FN (False Negatives): Incorrectly predicted negative
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        labels: List of class labels (optional)
        
    Returns:
        Dictionary with confusion matrix and TP/TN/FP/FN values
    """
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    result = {
        "confusion_matrix": cm,
        "labels": labels if labels is not None else np.unique(np.concatenate([y_true, y_pred])),
    }
    
    # For binary classification, extract TP, TN, FP, FN
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        result["TP"] = int(tp)
        result["TN"] = int(tn)
        result["FP"] = int(fp)
        result["FN"] = int(fn)
    else:
        # For multiclass, compute per-class TP, TN, FP, FN
        n_classes = cm.shape[0]
        per_class = {}
        
        for i in range(n_classes):
            tp = cm[i, i]
            fn = cm[i, :].sum() - tp
            fp = cm[:, i].sum() - tp
            tn = cm.sum() - tp - fn - fp
            
            class_label = result["labels"][i] if i < len(result["labels"]) else i
            per_class[class_label] = {
                "TP": int(tp),
                "TN": int(tn),
                "FP": int(fp),
                "FN": int(fn),
            }
        
        result["per_class"] = per_class
    
    return result


def compute_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                                   average: str = "weighted") -> dict:
    """
    Compute classification metrics: Accuracy, Precision, Recall, F1.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        average: Averaging method for multiclass ('micro', 'macro', 'weighted')
        
    Returns:
        Dictionary with all computed metrics
    """
    # Determine if binary or multiclass
    n_classes = len(np.unique(np.concatenate([y_true, y_pred])))
    
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
    }
    
    if n_classes == 2:
        # Binary classification
        metrics["precision"] = float(precision_score(y_true, y_pred, zero_division=0))
        metrics["recall"] = float(recall_score(y_true, y_pred, zero_division=0))
        metrics["f1"] = float(f1_score(y_true, y_pred, zero_division=0))
    else:
        # Multiclass - compute multiple averages
        for avg in ["micro", "macro", "weighted"]:
            metrics[f"precision_{avg}"] = float(precision_score(y_true, y_pred, average=avg, zero_division=0))
            metrics[f"recall_{avg}"] = float(recall_score(y_true, y_pred, average=avg, zero_division=0))
            metrics[f"f1_{avg}"] = float(f1_score(y_true, y_pred, average=avg, zero_division=0))
        
        # Also provide simple names using the specified average
        metrics["precision"] = metrics[f"precision_{average}"]
        metrics["recall"] = metrics[f"recall_{average}"]
        metrics["f1"] = metrics[f"f1_{average}"]
    
    return metrics


def compute_all_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Compute all classification metrics including per-class breakdown.
    
    This is the main function to use for evaluation.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        
    Returns:
        Comprehensive dictionary with all metrics
    """
    # Get unique labels
    labels = np.unique(np.concatenate([y_true, y_pred]))
    n_classes = len(labels)
    
    # Overall metrics
    overall = compute_classification_metrics(y_true, y_pred)
    
    # Confusion matrix
    cm_result = compute_confusion_matrix(y_true, y_pred)
    
    # Per-class metrics
    per_class = {}
    
    for label in labels:
        # Binary masks for this class
        y_true_binary = (y_true == label).astype(int)
        y_pred_binary = (y_pred == label).astype(int)
        
        tp = np.sum((y_true_binary == 1) & (y_pred_binary == 1))
        tn = np.sum((y_true_binary == 0) & (y_pred_binary == 0))
        fp = np.sum((y_true_binary == 0) & (y_pred_binary == 1))
        fn = np.sum((y_true_binary == 1) & (y_pred_binary == 0))
        
        # Calculate metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        per_class[label] = {
            "TP": int(tp),
            "TN": int(tn),
            "FP": int(fp),
            "FN": int(fn),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "support": int(np.sum(y_true == label)),  # Number of true instances
        }
    
    return {
        "overall": overall,
        "confusion_matrix": cm_result["confusion_matrix"],
        "per_class": per_class,
        "class_labels": list(labels),
        "n_classes": n_classes,
    }


def format_confusion_matrix_display(cm: np.ndarray, labels: list = None) -> str:
    """
    Format confusion matrix for display.
    
    Args:
        cm: Confusion matrix array
        labels: Class labels
        
    Returns:
        Formatted string representation
    """
    if labels is None:
        labels = [f"Class_{i}" for i in range(cm.shape[0])]
    
    # Header
    header = "         " + "  ".join([f"{l:>8}" for l in labels])
    
    # Rows
    rows = []
    for i, label in enumerate(labels):
        row_values = "  ".join([f"{cm[i, j]:>8}" for j in range(cm.shape[1])])
        rows.append(f"{label:>8} {row_values}")
    
    # Combine
    lines = [
        "Matrice de Confusion",
        "=" * len(header),
        "Predicted →",
        header,
        "-" * len(header),
    ] + rows
    
    return "\n".join(lines)


def generate_classification_report(y_true: np.ndarray, y_pred: np.ndarray,
                                   target_names: list = None) -> str:
    """
    Generate a full classification report.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        target_names: Optional list of class names
        
    Returns:
        Formatted classification report string
    """
    return classification_report(y_true, y_pred, target_names=target_names, zero_division=0)


def compute_precision_recall_curve_data(y_true: np.ndarray, 
                                         y_scores: np.ndarray,
                                         pos_label: Union[int, str] = 1) -> dict:
    """
    Compute data for precision-recall curve.
    
    Args:
        y_true: True binary labels
        y_scores: Probability scores
        pos_label: Positive class label
        
    Returns:
        Dictionary with precision, recall, and threshold values
    """
    from sklearn.metrics import precision_recall_curve, average_precision_score
    
    # Convert to binary if needed
    y_binary = (y_true == pos_label).astype(int)
    
    precision, recall, thresholds = precision_recall_curve(y_binary, y_scores)
    avg_precision = average_precision_score(y_binary, y_scores)
    
    return {
        "precision": precision,
        "recall": recall,
        "thresholds": thresholds,
        "average_precision": avg_precision,
    }
