"""
Main classification orchestrator.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd

from .knn import run_knn
from .naive_bayes import run_naive_bayes
from .decision_tree import run_c45
from .svm import run_svm
from .metrics import compute_all_metrics


@dataclass
class ClassificationResult:
    """Data class to hold classification results."""
    y_true: np.ndarray
    y_pred: np.ndarray
    params: dict
    timestamp: str
    algo: str
    metrics: dict
    confusion_matrix: np.ndarray
    class_labels: list = field(default_factory=list)
    model: Optional[object] = None


def train_test_split_data(X: pd.DataFrame, y: pd.Series, 
                          test_size: float = 0.2, 
                          random_state: int = 42) -> tuple:
    """
    Split data into training and test sets.
    
    Args:
        X: Feature DataFrame
        y: Target Series
        test_size: Proportion of data for testing (default: 0.2 = 20%)
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    from sklearn.model_selection import train_test_split
    
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)


def run_classification(algo_name: str, params: dict, 
                        X_train: np.ndarray, y_train: np.ndarray,
                        X_test: np.ndarray, y_test: np.ndarray) -> dict:
    """
    Execute the specified classification algorithm on the data.
    
    Args:
        algo_name: Name of the algorithm ('k-NN', 'Naive Bayes', 'C4.5', 'SVM')
        params: Dictionary of algorithm parameters
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        y_test: True test labels
        
    Returns:
        Dictionary containing predictions, metrics, and other results
    """
    result = {
        "y_true": y_test,
        "y_pred": None,
        "params": params,
        "timestamp": datetime.utcnow().isoformat(),
        "algo": algo_name,
        "model": None,
    }
    
    # Convert to numpy if needed
    if isinstance(X_train, pd.DataFrame):
        X_train = X_train.to_numpy()
    if isinstance(X_test, pd.DataFrame):
        X_test = X_test.to_numpy()
    if isinstance(y_train, pd.Series):
        y_train = y_train.to_numpy()
    if isinstance(y_test, pd.Series):
        y_test = y_test.to_numpy()
    
    # Route to appropriate algorithm
    if algo_name == "k-NN":
        algo_result = run_knn(X_train, y_train, X_test, params)
        result["y_pred"] = algo_result["predictions"]
        result["model"] = algo_result.get("model")
        
    elif algo_name == "Naive Bayes":
        algo_result = run_naive_bayes(X_train, y_train, X_test, params)
        result["y_pred"] = algo_result["predictions"]
        result["model"] = algo_result.get("model")
        result["class_probabilities"] = algo_result.get("probabilities")
        
    elif algo_name == "C4.5":
        algo_result = run_c45(X_train, y_train, X_test, params)
        result["y_pred"] = algo_result["predictions"]
        result["model"] = algo_result.get("model")
        result["feature_importances"] = algo_result.get("feature_importances")
        
    elif algo_name == "SVM":
        algo_result = run_svm(X_train, y_train, X_test, params)
        result["y_pred"] = algo_result["predictions"]
        result["model"] = algo_result.get("model")
        
    else:
        raise ValueError(f"Algorithme non supporté: {algo_name}")
    
    # Compute metrics
    metrics = compute_all_metrics(y_test, result["y_pred"])
    result["metrics"] = metrics["overall"]
    result["confusion_matrix"] = metrics["confusion_matrix"]
    result["per_class_metrics"] = metrics["per_class"]
    result["class_labels"] = metrics["class_labels"]
    
    return result


def compare_classifiers(X_train: np.ndarray, y_train: np.ndarray,
                        X_test: np.ndarray, y_test: np.ndarray,
                        algorithms: list = None) -> dict:
    """
    Compare multiple classification algorithms on the same data.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        y_test: True test labels
        algorithms: List of algorithm configs [{'name': str, 'params': dict}]
        
    Returns:
        Dictionary with comparison results
    """
    if algorithms is None:
        algorithms = [
            {"name": "k-NN", "params": {"k": 5}},
            {"name": "Naive Bayes", "params": {}},
            {"name": "C4.5", "params": {}},
            {"name": "SVM", "params": {"kernel": "rbf"}},
        ]
    
    results = {}
    for algo in algorithms:
        try:
            result = run_classification(
                algo["name"], 
                algo["params"],
                X_train, y_train,
                X_test, y_test
            )
            results[algo["name"]] = result
        except Exception as e:
            results[algo["name"]] = {"error": str(e)}
    
    return results
