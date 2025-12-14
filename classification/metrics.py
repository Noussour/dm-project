"""
Classification metrics utilities.

Custom implementation from scratch.
Computes evaluation metrics for classification:
- Confusion Matrix (TP, TN, FP, FN)
- Precision
- Recall
- F-measure (F1 score)
"""

import numpy as np
from typing import Union, List


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
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Get unique labels
    if labels is None:
        labels = np.unique(np.concatenate([y_true, y_pred]))
    labels = np.array(labels)
    
    n_labels = len(labels)
    
    # Create confusion matrix
    cm = np.zeros((n_labels, n_labels), dtype=int)
    
    for i, true_label in enumerate(labels):
        for j, pred_label in enumerate(labels):
            cm[i, j] = np.sum((y_true == true_label) & (y_pred == pred_label))
    
    result = {
        "confusion_matrix": cm,
        "labels": labels.tolist() if isinstance(labels, np.ndarray) else labels,
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
        per_class = {}
        
        for i in range(n_labels):
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


def accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute accuracy: (TP + TN) / total.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        
    Returns:
        Accuracy score
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return np.mean(y_true == y_pred)


def precision_score(y_true: np.ndarray, y_pred: np.ndarray,
                    average: str = "weighted", zero_division: int = 0) -> float:
    """
    Compute precision: TP / (TP + FP).
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        average: Averaging method ('micro', 'macro', 'weighted', 'binary')
        zero_division: Value to return when there's a zero division
        
    Returns:
        Precision score
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    labels = np.unique(np.concatenate([y_true, y_pred]))
    
    if len(labels) == 2 and average == "binary":
        # Binary case - assume positive class is labels[1]
        tp = np.sum((y_true == labels[1]) & (y_pred == labels[1]))
        fp = np.sum((y_true == labels[0]) & (y_pred == labels[1]))
        return tp / (tp + fp) if (tp + fp) > 0 else zero_division
    
    precisions = []
    supports = []
    
    for label in labels:
        tp = np.sum((y_true == label) & (y_pred == label))
        fp = np.sum((y_true != label) & (y_pred == label))
        
        prec = tp / (tp + fp) if (tp + fp) > 0 else zero_division
        precisions.append(prec)
        supports.append(np.sum(y_true == label))
    
    precisions = np.array(precisions)
    supports = np.array(supports)
    
    if average == "micro":
        total_tp = sum(np.sum((y_true == label) & (y_pred == label)) for label in labels)
        total_fp = sum(np.sum((y_true != label) & (y_pred == label)) for label in labels)
        return total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else zero_division
    elif average == "macro":
        return np.mean(precisions)
    elif average == "weighted":
        return np.average(precisions, weights=supports) if supports.sum() > 0 else zero_division
    else:
        return precisions


def recall_score(y_true: np.ndarray, y_pred: np.ndarray,
                 average: str = "weighted", zero_division: int = 0) -> float:
    """
    Compute recall: TP / (TP + FN).
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        average: Averaging method ('micro', 'macro', 'weighted', 'binary')
        zero_division: Value to return when there's a zero division
        
    Returns:
        Recall score
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    labels = np.unique(np.concatenate([y_true, y_pred]))
    
    if len(labels) == 2 and average == "binary":
        # Binary case
        tp = np.sum((y_true == labels[1]) & (y_pred == labels[1]))
        fn = np.sum((y_true == labels[1]) & (y_pred == labels[0]))
        return tp / (tp + fn) if (tp + fn) > 0 else zero_division
    
    recalls = []
    supports = []
    
    for label in labels:
        tp = np.sum((y_true == label) & (y_pred == label))
        fn = np.sum((y_true == label) & (y_pred != label))
        
        rec = tp / (tp + fn) if (tp + fn) > 0 else zero_division
        recalls.append(rec)
        supports.append(np.sum(y_true == label))
    
    recalls = np.array(recalls)
    supports = np.array(supports)
    
    if average == "micro":
        total_tp = sum(np.sum((y_true == label) & (y_pred == label)) for label in labels)
        total_fn = sum(np.sum((y_true == label) & (y_pred != label)) for label in labels)
        return total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else zero_division
    elif average == "macro":
        return np.mean(recalls)
    elif average == "weighted":
        return np.average(recalls, weights=supports) if supports.sum() > 0 else zero_division
    else:
        return recalls


def f1_score(y_true: np.ndarray, y_pred: np.ndarray,
             average: str = "weighted", zero_division: int = 0) -> float:
    """
    Compute F1 score: 2 * (precision * recall) / (precision + recall).
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        average: Averaging method ('micro', 'macro', 'weighted', 'binary')
        zero_division: Value to return when there's a zero division
        
    Returns:
        F1 score
    """
    prec = precision_score(y_true, y_pred, average=average, zero_division=zero_division)
    rec = recall_score(y_true, y_pred, average=average, zero_division=zero_division)
    
    if isinstance(prec, np.ndarray):
        f1s = []
        for p, r in zip(prec, rec):
            f1s.append(2 * p * r / (p + r) if (p + r) > 0 else zero_division)
        return np.array(f1s)
    else:
        return 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else zero_division


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
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Determine if binary or multiclass
    n_classes = len(np.unique(np.concatenate([y_true, y_pred])))
    
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
    }
    
    if n_classes == 2:
        # Binary classification
        metrics["precision"] = float(precision_score(y_true, y_pred, average="binary", zero_division=0))
        metrics["recall"] = float(recall_score(y_true, y_pred, average="binary", zero_division=0))
        metrics["f1"] = float(f1_score(y_true, y_pred, average="binary", zero_division=0))
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
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
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
        "Confusion Matrix",
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
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    labels = np.unique(np.concatenate([y_true, y_pred]))
    
    if target_names is None:
        target_names = [str(label) for label in labels]
    
    # Header
    lines = [
        "              precision    recall  f1-score   support",
        ""
    ]
    
    # Per-class metrics
    total_support = 0
    weighted_precision = 0
    weighted_recall = 0
    weighted_f1 = 0
    
    for i, (label, name) in enumerate(zip(labels, target_names)):
        y_true_binary = (y_true == label).astype(int)
        y_pred_binary = (y_pred == label).astype(int)
        
        tp = np.sum((y_true_binary == 1) & (y_pred_binary == 1))
        fp = np.sum((y_true_binary == 0) & (y_pred_binary == 1))
        fn = np.sum((y_true_binary == 1) & (y_pred_binary == 0))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        support = np.sum(y_true == label)
        
        lines.append(f"{name:>12}       {precision:.2f}      {recall:.2f}      {f1:.2f}       {support}")
        
        total_support += support
        weighted_precision += precision * support
        weighted_recall += recall * support
        weighted_f1 += f1 * support
    
    # Averages
    lines.append("")
    acc = accuracy_score(y_true, y_pred)
    lines.append(f"    accuracy                           {acc:.2f}       {total_support}")
    
    if total_support > 0:
        weighted_precision /= total_support
        weighted_recall /= total_support
        weighted_f1 /= total_support
    
    lines.append(f"   macro avg       {np.mean([l for l in precision_score(y_true, y_pred, average=None)]):.2f}      {np.mean([l for l in recall_score(y_true, y_pred, average=None)]):.2f}      {np.mean([l for l in f1_score(y_true, y_pred, average=None)]):.2f}       {total_support}")
    lines.append(f"weighted avg       {weighted_precision:.2f}      {weighted_recall:.2f}      {weighted_f1:.2f}       {total_support}")
    
    return "\n".join(lines)


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
    y_true = np.array(y_true)
    y_scores = np.array(y_scores)
    
    # Convert to binary if needed
    y_binary = (y_true == pos_label).astype(int)
    
    # Sort by score descending
    sorted_indices = np.argsort(y_scores)[::-1]
    y_scores_sorted = y_scores[sorted_indices]
    y_true_sorted = y_binary[sorted_indices]
    
    # Get unique thresholds
    thresholds = np.unique(y_scores_sorted)[::-1]
    
    precisions = [1.0]
    recalls = [0.0]
    
    for threshold in thresholds:
        y_pred = (y_scores >= threshold).astype(int)
        
        tp = np.sum((y_binary == 1) & (y_pred == 1))
        fp = np.sum((y_binary == 0) & (y_pred == 1))
        fn = np.sum((y_binary == 1) & (y_pred == 0))
        
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        precisions.append(prec)
        recalls.append(rec)
    
    precisions = np.array(precisions)
    recalls = np.array(recalls)
    
    # Compute average precision
    avg_precision = np.sum((recalls[1:] - recalls[:-1]) * precisions[1:])
    
    return {
        "precision": precisions,
        "recall": recalls,
        "thresholds": thresholds,
        "average_precision": float(avg_precision),
    }
