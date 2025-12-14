"""
k-Nearest Neighbors (k-NN) classifier implementation.

Custom implementation from scratch.
k-NN is a non-parametric, instance-based learning algorithm that classifies
new data points based on the majority class of their k nearest neighbors.
"""

import numpy as np
from typing import Optional
from collections import Counter


def _euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Compute Euclidean distance between two points."""
    return np.sqrt(np.sum((a - b) ** 2))


def _manhattan_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Compute Manhattan distance between two points."""
    return np.sum(np.abs(a - b))


def _compute_distances(X_train: np.ndarray, x_test: np.ndarray, 
                       metric: str = "euclidean") -> np.ndarray:
    """
    Compute distances from a test point to all training points.
    
    Args:
        X_train: Training features (n_samples, n_features)
        x_test: Single test point (n_features,)
        metric: Distance metric ('euclidean' or 'manhattan')
        
    Returns:
        Array of distances to each training point
    """
    n_samples = X_train.shape[0]
    distances = np.zeros(n_samples)
    
    dist_func = _euclidean_distance if metric == "euclidean" else _manhattan_distance
    
    for i in range(n_samples):
        distances[i] = dist_func(X_train[i], x_test)
    
    return distances


class KNNClassifier:
    """
    Custom k-Nearest Neighbors Classifier.
    
    Attributes:
        k: Number of neighbors
        metric: Distance metric
        weights: Weight function ('uniform' or 'distance')
        X_train: Stored training features
        y_train: Stored training labels
    """
    
    def __init__(self, n_neighbors: int = 5, metric: str = "euclidean", 
                 weights: str = "uniform"):
        """
        Initialize k-NN classifier.
        
        Args:
            n_neighbors: Number of neighbors to use
            metric: Distance metric ('euclidean', 'manhattan')
            weights: Weight function ('uniform', 'distance')
        """
        self.k = n_neighbors
        self.metric = metric
        self.weights = weights
        self.X_train = None
        self.y_train = None
        self.classes_ = None
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fit the k-NN classifier (store training data).
        
        Args:
            X: Training features
            y: Training labels
        """
        self.X_train = np.array(X)
        self.y_train = np.array(y)
        self.classes_ = np.unique(y)
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels for test samples.
        
        Args:
            X: Test features
            
        Returns:
            Predicted class labels
        """
        X = np.array(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        predictions = []
        for x_test in X:
            prediction = self._predict_single(x_test)
            predictions.append(prediction)
        
        return np.array(predictions)
    
    def _predict_single(self, x_test: np.ndarray):
        """Predict class for a single test point."""
        # Compute distances to all training points
        distances = _compute_distances(self.X_train, x_test, self.metric)
        
        # Get indices of k nearest neighbors
        k_indices = np.argsort(distances)[:self.k]
        k_labels = self.y_train[k_indices]
        k_distances = distances[k_indices]
        
        if self.weights == "uniform":
            # Simple majority voting
            label_counts = Counter(k_labels)
            return label_counts.most_common(1)[0][0]
        else:
            # Distance-weighted voting
            weights = np.zeros(len(self.classes_))
            for i, label in enumerate(k_labels):
                # Avoid division by zero
                weight = 1.0 / (k_distances[i] + 1e-10)
                class_idx = np.where(self.classes_ == label)[0][0]
                weights[class_idx] += weight
            return self.classes_[np.argmax(weights)]
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities for test samples.
        
        Args:
            X: Test features
            
        Returns:
            Class probabilities for each sample
        """
        X = np.array(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        probabilities = []
        for x_test in X:
            proba = self._predict_proba_single(x_test)
            probabilities.append(proba)
        
        return np.array(probabilities)
    
    def _predict_proba_single(self, x_test: np.ndarray) -> np.ndarray:
        """Compute class probabilities for a single test point."""
        # Compute distances to all training points
        distances = _compute_distances(self.X_train, x_test, self.metric)
        
        # Get indices of k nearest neighbors
        k_indices = np.argsort(distances)[:self.k]
        k_labels = self.y_train[k_indices]
        k_distances = distances[k_indices]
        
        # Compute probabilities based on voting
        proba = np.zeros(len(self.classes_))
        
        if self.weights == "uniform":
            for label in k_labels:
                class_idx = np.where(self.classes_ == label)[0][0]
                proba[class_idx] += 1.0
            proba /= self.k
        else:
            total_weight = 0
            for i, label in enumerate(k_labels):
                weight = 1.0 / (k_distances[i] + 1e-10)
                class_idx = np.where(self.classes_ == label)[0][0]
                proba[class_idx] += weight
                total_weight += weight
            proba /= total_weight
        
        return proba
    
    def kneighbors(self, X: np.ndarray) -> tuple:
        """
        Find k nearest neighbors for test samples.
        
        Args:
            X: Test features
            
        Returns:
            Tuple of (distances, indices) arrays
        """
        X = np.array(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        all_distances = []
        all_indices = []
        
        for x_test in X:
            distances = _compute_distances(self.X_train, x_test, self.metric)
            k_indices = np.argsort(distances)[:self.k]
            k_distances = distances[k_indices]
            all_distances.append(k_distances)
            all_indices.append(k_indices)
        
        return np.array(all_distances), np.array(all_indices)


def run_knn(X_train: np.ndarray, y_train: np.ndarray, 
            X_test: np.ndarray, params: dict) -> dict:
    """
    Execute k-NN classification using custom implementation.
    
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
    knn = KNNClassifier(n_neighbors=k, metric=metric, weights=weights)
    knn.fit(X_train, y_train)
    
    # Predict
    predictions = knn.predict(X_test)
    
    # Get probabilities
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
        
        # Store results - use 'precision' which is always available
        results["accuracy"].append(metrics["overall"]["accuracy"])
        results["precision"].append(metrics["overall"].get("precision", metrics["overall"].get("precision_macro", 0)))
        results["recall"].append(metrics["overall"].get("recall", metrics["overall"].get("recall_macro", 0)))
        results["f1"].append(metrics["overall"].get("f1", metrics["overall"].get("f1_macro", 0)))
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


def predict_with_neighbors(knn_model: KNNClassifier, 
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
