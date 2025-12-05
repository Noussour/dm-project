"""
Clustering metrics utilities.

Implements standard internal clustering validation metrics:
- Silhouette Score: Measures cluster cohesion and separation
- Calinski-Harabasz Index: Ratio of between-cluster to within-cluster dispersion  
- Davies-Bouldin Index: Average similarity between clusters
- Inertia/WCSS: Within-Cluster Sum of Squares (for partition methods)
"""

import numpy as np
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
)


def n_clusters_from_labels(labels: np.ndarray) -> int:
    """
    Return the number of clusters (excluding noise label -1 for DBSCAN).
    
    Args:
        labels: Array of cluster labels
        
    Returns:
        Number of unique clusters
    """
    if labels is None or len(labels) == 0:
        return 0
    
    unique = np.unique(labels)
    
    # For DBSCAN, -1 is noise
    if -1 in unique:
        return len(unique) - 1
    
    return len(unique)


def compute_inertia(X: np.ndarray, labels: np.ndarray) -> float:
    """
    Compute inertia (WCSS - Within Cluster Sum of Squares).
    
    This is the sum of squared distances from each point to its cluster centroid.
    Lower values indicate tighter clusters.
    
    Args:
        X: Feature array
        labels: Cluster labels
        
    Returns:
        Inertia value
    """
    unique_labels = np.unique(labels)
    # Exclude noise points (label -1)
    unique_labels = unique_labels[unique_labels >= 0]
    
    inertia = 0.0
    for label in unique_labels:
        cluster_points = X[labels == label]
        if len(cluster_points) > 0:
            centroid = np.mean(cluster_points, axis=0)
            inertia += np.sum((cluster_points - centroid) ** 2)
    
    return float(inertia)


def compute_metrics(X: np.ndarray, labels: np.ndarray) -> dict:
    """
    Compute clustering evaluation metrics.
    
    Calculates:
    - Silhouette Score: [-1, 1], higher is better
    - Calinski-Harabasz Index: higher is better
    - Davies-Bouldin Index: lower is better
    - Inertia (WCSS): lower is better (for partition-based methods)
    
    Args:
        X: Feature array
        labels: Cluster labels
        
    Returns:
        Dictionary with metric names as keys and scores as values
    """
    metrics = {
        "silhouette": None,
        "calinski_harabasz": None,
        "davies_bouldin": None,
        "inertia": None,
    }
    
    n_clusters = n_clusters_from_labels(labels)
    n_samples = X.shape[0]
    
    # Silhouette requires 2 <= n_clusters <= n_samples - 1
    if n_clusters >= 2 and n_clusters < n_samples:
        try:
            metrics["silhouette"] = float(silhouette_score(X, labels))
        except Exception:
            pass
    
    # Calinski-Harabasz and Davies-Bouldin require at least 2 clusters
    if n_clusters >= 2:
        try:
            metrics["calinski_harabasz"] = float(calinski_harabasz_score(X, labels))
        except Exception:
            pass
        
        try:
            metrics["davies_bouldin"] = float(davies_bouldin_score(X, labels))
        except Exception:
            pass
    
    # Compute inertia
    if n_clusters >= 1:
        try:
            metrics["inertia"] = compute_inertia(X, labels)
        except Exception:
            pass
    
    return metrics
