"""
K-Medoids (PAM - Partitioning Around Medoids) clustering implementation.

K-Medoids is similar to K-Means but uses actual data points (medoids) as cluster centers
instead of calculated means (centroids). This makes it more robust to outliers.

Custom implementation from scratch - no sklearn_extra or scipy dependency.
"""

import numpy as np


def _euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Compute Euclidean distance between two points."""
    return np.sqrt(np.sum((a - b) ** 2))


def _manhattan_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Compute Manhattan distance between two points."""
    return np.sum(np.abs(a - b))


def _compute_distance_matrix(X: np.ndarray, metric: str = "euclidean") -> np.ndarray:
    """
    Compute pairwise distance matrix.
    
    Args:
        X: Data array of shape (n_samples, n_features)
        metric: Distance metric ('euclidean' or 'manhattan')
        
    Returns:
        Distance matrix of shape (n_samples, n_samples)
    """
    n = X.shape[0]
    dist_matrix = np.zeros((n, n))
    
    dist_func = _euclidean_distance if metric == "euclidean" else _manhattan_distance
    
    for i in range(n):
        for j in range(i + 1, n):
            d = dist_func(X[i], X[j])
            dist_matrix[i, j] = d
            dist_matrix[j, i] = d
    
    return dist_matrix


def run_kmedoids(X: np.ndarray, params: dict) -> dict:
    """
    Execute K-Medoids clustering using PAM algorithm.
    
    Args:
        X: Feature array
        params: Dictionary with parameters:
            - 'n_clusters': Number of clusters (default: 3)
            - 'metric': Distance metric ('euclidean', 'manhattan') (default: 'euclidean')
            - 'max_iter': Maximum iterations (default: 300)
        
    Returns:
        Dictionary with labels, medoids, and inertia
    """
    n_clusters = int(params.get("n_clusters", 3))
    metric = params.get("metric", "euclidean")
    max_iter = int(params.get("max_iter", 300))
    
    # Run custom K-Medoids
    labels, medoid_indices, inertia = kmedoids_pam(
        X, n_clusters, metric=metric, max_iter=max_iter, random_state=42
    )
    
    # Get medoid points
    medoids = X[medoid_indices]
    
    return {
        "labels": labels,
        "medoids": medoids,
        "medoid_indices": medoid_indices,
        "inertia": float(inertia),
    }


def kmedoids_pam(X: np.ndarray, n_clusters: int, metric: str = "euclidean",
                  max_iter: int = 300, random_state: int = None) -> tuple:
    """
    K-Medoids clustering using the PAM (Partitioning Around Medoids) algorithm.
    
    Args:
        X: Feature array (n_samples, n_features)
        n_clusters: Number of clusters
        metric: Distance metric
        max_iter: Maximum iterations
        random_state: Random seed
        
    Returns:
        Tuple of (labels, medoid_indices, inertia)
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    n_samples = X.shape[0]
    
    # Compute pairwise distance matrix
    D = _compute_distance_matrix(X, metric)
    
    # Initialize medoids using k-medoids++ style initialization
    medoid_indices = _kmedoids_plusplus_init(D, n_clusters)
    
    # PAM algorithm
    for iteration in range(max_iter):
        # Assign points to nearest medoid
        labels = np.argmin(D[:, medoid_indices], axis=1)
        
        # Update medoids
        new_medoid_indices = np.zeros(n_clusters, dtype=int)
        
        for k in range(n_clusters):
            cluster_mask = labels == k
            cluster_indices = np.where(cluster_mask)[0]
            
            if len(cluster_indices) == 0:
                # Empty cluster - keep old medoid
                new_medoid_indices[k] = medoid_indices[k]
                continue
            
            # Find the point that minimizes total distance within cluster
            best_medoid = cluster_indices[0]
            best_cost = float('inf')
            
            for candidate in cluster_indices:
                cost = sum(D[candidate, other] for other in cluster_indices)
                if cost < best_cost:
                    best_cost = cost
                    best_medoid = candidate
            
            new_medoid_indices[k] = best_medoid
        
        # Check for convergence
        if np.array_equal(medoid_indices, new_medoid_indices):
            break
        
        medoid_indices = new_medoid_indices
    
    # Final assignment
    labels = np.argmin(D[:, medoid_indices], axis=1)
    
    # Compute inertia (sum of distances to medoids)
    inertia = sum(D[i, medoid_indices[labels[i]]] for i in range(n_samples))
    
    return labels, medoid_indices, inertia


def _kmedoids_plusplus_init(D: np.ndarray, n_clusters: int) -> np.ndarray:
    """
    Initialize medoids using k-medoids++ method (similar to k-means++).
    
    Args:
        D: Pairwise distance matrix
        n_clusters: Number of clusters
        
    Returns:
        Array of medoid indices
    """
    n_samples = D.shape[0]
    medoid_indices = np.zeros(n_clusters, dtype=int)
    
    # Choose first medoid randomly
    medoid_indices[0] = np.random.randint(n_samples)
    
    # Choose remaining medoids
    for k in range(1, n_clusters):
        # Compute distance to nearest existing medoid for each point
        min_distances = D[:, medoid_indices[:k]].min(axis=1)
        
        # Square distances for probability weighting
        probabilities = min_distances ** 2
        prob_sum = probabilities.sum()
        if prob_sum > 0:
            probabilities /= prob_sum
        else:
            probabilities = np.ones(n_samples) / n_samples
        
        # Choose next medoid with probability proportional to distance squared
        medoid_indices[k] = np.random.choice(n_samples, p=probabilities)
    
    return medoid_indices


def compute_kmedoids_elbow(X: np.ndarray, k_range: range = range(2, 11), 
                           metric: str = "euclidean") -> tuple:
    """
    Compute inertia values for elbow method with K-Medoids.
    
    Args:
        X: Feature array
        k_range: Range of k values to test
        metric: Distance metric to use
        
    Returns:
        Tuple of (k_values, inertias)
    """
    k_values = list(k_range)
    inertias = []
    
    for k in k_values:
        _, _, inertia = kmedoids_pam(X, k, metric=metric, random_state=42)
        inertias.append(inertia)
    
    return k_values, inertias
