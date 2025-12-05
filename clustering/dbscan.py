"""
DBSCAN clustering implementation.

Custom implementation from scratch based on TP3 (Data Mining course).
No sklearn or scipy dependency.
"""

import numpy as np


def _euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Compute Euclidean distance between two points."""
    return np.sqrt(np.sum((a - b) ** 2))


def _compute_distance_matrix(X: np.ndarray) -> np.ndarray:
    """
    Compute pairwise Euclidean distance matrix.
    
    Args:
        X: Data array of shape (n_samples, n_features)
        
    Returns:
        Distance matrix of shape (n_samples, n_samples)
    """
    n = X.shape[0]
    dist_matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(i + 1, n):
            d = _euclidean_distance(X[i], X[j])
            dist_matrix[i, j] = d
            dist_matrix[j, i] = d
    
    return dist_matrix


def dbscan_custom(X: np.ndarray, eps: float = 0.5, min_samples: int = 5) -> tuple:
    """
    Custom DBSCAN implementation.
    
    DBSCAN (Density-Based Spatial Clustering of Applications with Noise)
    identifies clusters as dense regions of points separated by regions of low density.
    
    Algorithm:
    1. For each point, find all neighbors within eps radius
    2. If point has >= min_samples neighbors, it's a core point
    3. Create clusters by connecting core points that are neighbors
    4. Non-core points within eps of a core point are border points
    5. Points not reachable from any core point are noise
    
    Args:
        X: Feature array (n_samples, n_features)
        eps: Maximum distance between two samples for neighborhood
        min_samples: Minimum number of samples in neighborhood for core point
        
    Returns:
        Tuple of (labels, core_sample_mask)
        - labels: cluster labels (-1 for noise)
        - core_sample_mask: boolean array indicating core points
    """
    n_samples = X.shape[0]
    
    # Calculate pairwise distances
    distances = _compute_distance_matrix(X)
    
    # Find neighbors within eps for each point
    neighbors = [np.where(distances[i] <= eps)[0] for i in range(n_samples)]
    
    # Identify core points (points with >= min_samples neighbors)
    core_sample_mask = np.array([len(neighbors[i]) >= min_samples for i in range(n_samples)])
    
    # Initialize labels: -2 = unvisited, -1 = noise, >= 0 = cluster id
    labels = np.full(n_samples, -2, dtype=int)
    
    cluster_id = -1
    
    for i in range(n_samples):
        if labels[i] != -2:  # Already processed
            continue
        
        if not core_sample_mask[i]:  # Not a core point
            labels[i] = -1  # Mark as noise (may become border point later)
            continue
        
        # Start new cluster
        cluster_id += 1
        labels[i] = cluster_id
        
        # Initialize seed set with neighbors of current point
        seeds = set(neighbors[i].tolist())
        seeds.discard(i)  # Remove current point
        
        # Expand cluster
        while seeds:
            current = seeds.pop()
            
            if labels[current] == -1:
                # Noise point becomes border point
                labels[current] = cluster_id
            
            if labels[current] != -2:  # Already processed
                continue
            
            labels[current] = cluster_id
            
            # If current point is core, add its unvisited neighbors to seeds
            if core_sample_mask[current]:
                for neighbor in neighbors[current]:
                    if labels[neighbor] == -2:
                        seeds.add(int(neighbor))
    
    return labels, core_sample_mask


def run_dbscan(X: np.ndarray, params: dict) -> dict:
    """
    Execute DBSCAN clustering using custom implementation.
    
    Args:
        X: Feature array
        params: Dictionary with 'eps' and 'min_samples' parameters
        
    Returns:
        Dictionary with labels and core_sample_mask
    """
    eps = float(params.get("eps", 0.5))
    min_samples = int(params.get("min_samples", 5))
    
    labels, core_sample_mask = dbscan_custom(X, eps=eps, min_samples=min_samples)
    
    return {
        "labels": labels,
        "core_sample_mask": core_sample_mask,
    }


def compute_kdistances(X: np.ndarray, k: int = 4) -> np.ndarray:
    """
    Compute k-distances for DBSCAN epsilon selection.
    
    The k-distance graph helps determine an appropriate eps value.
    Plot the k-distance for each point (sorted) and look for the "elbow" point.
    
    Args:
        X: Feature array
        k: Number of neighbors (typically equals min_samples)
        
    Returns:
        Array of k-distances sorted in descending order
    """
    n_samples = X.shape[0]
    
    # Ensure k doesn't exceed n_samples - 1
    k = min(k, n_samples - 1)
    
    if k <= 0:
        return np.array([0.0])
    
    # Calculate pairwise distances
    distances = _compute_distance_matrix(X)
    
    # For each point, find the k-th nearest neighbor distance
    k_distances = np.zeros(n_samples)
    for i in range(n_samples):
        # Sort distances and get k-th nearest (index k because index 0 is itself with distance 0)
        sorted_dists = np.sort(distances[i])
        k_distances[i] = sorted_dists[k] if k < len(sorted_dists) else sorted_dists[-1]
    
    # Sort in descending order for the k-distance graph
    return np.sort(k_distances)[::-1]

