"""
k-Nearest Neighbors (k-NN) classifier implementation.

k-NN is a non-parametric, instance-based learning algorithm that classifies
new data points based on the majority class of their k nearest neighbors.
"""

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from typing import Optional


def run_knn(X_train: np.ndarray, y_train: np.ndarray, 
            X_test: np.ndarray, params: dict) -> dict:
    """
    Execute k-NN classification.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        params: Dictionary with parameters:
            - 'k' or 'n_neighbors': Number of neighbors (default: 5)
            - 'metric': Distance metric (default: 'euclidean')
            - 'weights': Weight function ('uniform', 'distance') (default: 'uniform')
        
    Returns:
        Dictionary with predictions and model
    """
    # Get parameters
    k = int(params.get("k", params.get("n_neighbors", 5)))
    metric = params.get("metric", "euclidean")
    weights = params.get("weights", "uniform")
    
    # Create and train classifier
    knn = KNeighborsClassifier(
        n_neighbors=k,
        metric=metric,
        weights=weights,
    )
    knn.fit(X_train, y_train)
    
    # Predict
    predictions = knn.predict(X_test)
    
    # Get probabilities if available
    try:
        probabilities = knn.predict_proba(X_test)
    except Exception:
        probabilities = None
    
    return {
        "predictions": predictions,
        "model": knn,
        "probabilities": probabilities,
        "k": k,
    }


def evaluate_knn_k_range(X_train: np.ndarray, y_train: np.ndarray,
                         X_test: np.ndarray, y_test: np.ndarray,
                         k_range: range = range(1, 11),
                         metric: str = "euclidean") -> dict:
    """
    Evaluate k-NN for multiple values of k (1 to 10).
    
    This function is essential for finding the optimal k value:
    - Running k-NN with k = 1 to 10
    - Computing metrics for each k
    - Finding the optimal k
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        y_test: True test labels
        k_range: Range of k values to test (default: 1-10)
        metric: Distance metric
        
    Returns:
        Dictionary with evaluation results for each k
    """
    from .metrics import compute_all_metrics
    
    results = {
        "k_values": list(k_range),
        "accuracy": [],
        "precision": [],
        "recall": [],
        "f1": [],
        "confusion_matrices": [],
        "per_k_results": {},
    }
    
    for k in k_range:
        # Run k-NN
        knn_result = run_knn(X_train, y_train, X_test, {"k": k, "metric": metric})
        predictions = knn_result["predictions"]
        
        # Compute metrics
        metrics = compute_all_metrics(y_test, predictions)
        
        # Store results
        results["accuracy"].append(metrics["overall"]["accuracy"])
        results["precision"].append(metrics["overall"]["precision_macro"])
        results["recall"].append(metrics["overall"]["recall_macro"])
        results["f1"].append(metrics["overall"]["f1_macro"])
        results["confusion_matrices"].append(metrics["confusion_matrix"])
        
        results["per_k_results"][k] = {
            "predictions": predictions,
            "metrics": metrics,
        }
    
    # Find best k (based on accuracy or F1 score)
    best_k_idx = np.argmax(results["accuracy"])
    results["best_k"] = results["k_values"][best_k_idx]
    results["best_accuracy"] = results["accuracy"][best_k_idx]
    
    # Also find best k by F1
    best_k_f1_idx = np.argmax(results["f1"])
    results["best_k_f1"] = results["k_values"][best_k_f1_idx]
    results["best_f1"] = results["f1"][best_k_f1_idx]
    
    return results


def predict_with_neighbors(knn_model: KNeighborsClassifier, 
                            X_test: np.ndarray) -> tuple:
    """
    Get predictions along with neighbor information.
    
    Args:
        knn_model: Trained k-NN model
        X_test: Test features
        
    Returns:
        Tuple of (predictions, distances, neighbor_indices)
    """
    predictions = knn_model.predict(X_test)
    distances, indices = knn_model.kneighbors(X_test)
    
    return predictions, distances, indices
