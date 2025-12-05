"""
K-Means clustering implementation.

Custom implementation based on TP2 (Data Mining course).
"""

import numpy as np
from sklearn.metrics import silhouette_score


def kmeans_custom(X: np.ndarray, k: int, max_iters: int = 100, 
                  init: str = "k-means++", tolerance: float = 1e-4,
                  random_state: int = None) -> tuple:
    """
    Custom K-Means implementation.
    
    Based on TP2 implementation with improvements:
    - k-means++ initialization
    - Convergence tolerance check
    - Empty cluster handling
    
    Args:
        X: Feature array (n_samples, n_features)
        k: Number of clusters
        max_iters: Maximum iterations
        init: Initialization method ('k-means++' or 'random')
        tolerance: Convergence tolerance
        random_state: Random seed
        
    Returns:
        Tuple of (labels, centroids, inertia)
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    n_samples, n_features = X.shape
    
    # Initialize centroids
    if init == "k-means++":
        centroids = _kmeans_plusplus_init(X, k)
    else:  # random
        indices = np.random.choice(n_samples, size=k, replace=False)
        centroids = X[indices].copy()
    
    labels = np.zeros(n_samples, dtype=int)
    
    for iteration in range(max_iters):
        # Assignment step: assign each point to nearest centroid
        # Calculate distances from each point to each centroid
        distances = np.sqrt(((X[:, np.newaxis, :] - centroids[np.newaxis, :, :]) ** 2).sum(axis=2))
        labels = np.argmin(distances, axis=1)
        
        # Update step: recalculate centroids
        new_centroids = np.zeros_like(centroids)
        for i in range(k):
            cluster_points = X[labels == i]
            if len(cluster_points) > 0:
                new_centroids[i] = cluster_points.mean(axis=0)
            else:
                # Empty cluster: reinitialize with random point
                new_centroids[i] = X[np.random.randint(n_samples)]
        
        # Check convergence
        if np.all(np.abs(centroids - new_centroids) < tolerance):
            break
        
        centroids = new_centroids
    
    # Calculate final inertia (WCSS - Within Cluster Sum of Squares)
    inertia = 0.0
    for i in range(k):
        cluster_points = X[labels == i]
        if len(cluster_points) > 0:
            inertia += np.sum((cluster_points - centroids[i]) ** 2)
    
    return labels, centroids, inertia


def _kmeans_plusplus_init(X: np.ndarray, k: int) -> np.ndarray:
    """
    K-means++ initialization for better initial centroids.
    
    Algorithm:
    1. Choose first centroid uniformly at random
    2. For each remaining centroid:
       - Compute distance D(x) from each point to nearest existing centroid
       - Choose next centroid with probability proportional to D(x)^2
    
    Args:
        X: Feature array
        k: Number of clusters
        
    Returns:
        Array of initial centroids
    """
    n_samples = X.shape[0]
    centroids = np.zeros((k, X.shape[1]))
    
    # First centroid: random point
    centroids[0] = X[np.random.randint(n_samples)]
    
    for i in range(1, k):
        # Compute squared distances to nearest centroid
        distances = np.zeros(n_samples)
        for j in range(n_samples):
            min_dist = float('inf')
            for c in range(i):
                dist = np.sum((X[j] - centroids[c]) ** 2)
                if dist < min_dist:
                    min_dist = dist
            distances[j] = min_dist
        
        # Choose next centroid with probability proportional to distance squared
        probabilities = distances / distances.sum()
        next_idx = np.random.choice(n_samples, p=probabilities)
        centroids[i] = X[next_idx]
    
    return centroids


def run_kmeans(X: np.ndarray, params: dict) -> dict:
    """
    Execute K-Means clustering using custom implementation.
    
    Args:
        X: Feature array
        params: Dictionary with 'n_clusters' and 'init' parameters
        
    Returns:
        Dictionary with labels, centroids, and inertia
    """
    n_clusters = int(params.get("n_clusters", 3))
    init_method = params.get("init", "k-means++")
    
    # Run custom K-Means with multiple initializations to get best result
    best_labels = None
    best_centroids = None
    best_inertia = float('inf')
    
    n_init = 10  # Number of initializations
    for _ in range(n_init):
        labels, centroids, inertia = kmeans_custom(
            X, n_clusters, init=init_method, random_state=None
        )
        if inertia < best_inertia:
            best_inertia = inertia
            best_labels = labels
            best_centroids = centroids
    
    return {
        "labels": best_labels,
        "centroids": best_centroids,
        "inertia": float(best_inertia),
    }


def compute_elbow(X: np.ndarray, k_range: range = range(2, 11)) -> tuple[list, list]:
    """
    Compute inertia values for elbow method using custom K-Means.
    
    Args:
        X: Feature array
        k_range: Range of k values to test
        
    Returns:
        Tuple of (k_values, inertias)
    """
    k_values = list(k_range)
    inertias = []
    
    for k in k_values:
        # Run multiple times and take best inertia
        best_inertia = float('inf')
        for _ in range(5):
            _, _, inertia = kmeans_custom(X, k, random_state=None)
            if inertia < best_inertia:
                best_inertia = inertia
        inertias.append(best_inertia)
    
    return k_values, inertias


def compute_silhouette_scores(X: np.ndarray, k_range: range = range(2, 11)) -> tuple[list, list]:
    """
    Compute silhouette scores for different k values using custom K-Means.
    
    Args:
        X: Feature array
        k_range: Range of k values to test
        
    Returns:
        Tuple of (k_values, silhouette_scores)
    """
    k_values = list(k_range)
    scores = []
    
    for k in k_values:
        # Run multiple times and take best result
        best_score = -1
        for _ in range(5):
            labels, _, _ = kmeans_custom(X, k, random_state=None)
            if len(np.unique(labels)) >= 2:
                score = silhouette_score(X, labels)
                if score > best_score:
                    best_score = score
        scores.append(best_score)
    
    return k_values, scores

