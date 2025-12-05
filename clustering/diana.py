"""
DIANA (DIvisive ANAlysis) - Top-down hierarchical clustering implementation.

DIANA is a divisive (top-down) approach where all observations start in one cluster,
and splits are performed recursively as one moves down the hierarchy.

Custom implementation from scratch - no sklearn dependency.
"""

import numpy as np


def _euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Compute Euclidean distance between two points."""
    return np.sqrt(np.sum((a - b) ** 2))


def _manhattan_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Compute Manhattan distance between two points."""
    return np.sum(np.abs(a - b))


def _compute_pairwise_distances(points: np.ndarray, metric: str = "euclidean") -> np.ndarray:
    """
    Compute pairwise distance matrix.
    
    Args:
        points: Data array of shape (n_points, n_features)
        metric: Distance metric ('euclidean' or 'manhattan')
        
    Returns:
        Distance matrix of shape (n_points, n_points)
    """
    n = points.shape[0]
    dist_matrix = np.zeros((n, n))
    
    dist_func = _euclidean_distance if metric == "euclidean" else _manhattan_distance
    
    for i in range(n):
        for j in range(i + 1, n):
            d = dist_func(points[i], points[j])
            dist_matrix[i, j] = d
            dist_matrix[j, i] = d
    
    return dist_matrix


def run_diana(X: np.ndarray, params: dict) -> dict:
    """
    Execute DIANA (Divisive Analysis) hierarchical clustering.
    
    Args:
        X: Feature array
        params: Dictionary with parameters:
            - 'n_clusters': Number of clusters (default: 2)
            - 'metric': Distance metric ('euclidean', 'manhattan') (default: 'euclidean')
        
    Returns:
        Dictionary with labels and division history
    """
    n_clusters = int(params.get("n_clusters", 2))
    metric = params.get("metric", "euclidean")
    
    # Use custom DIANA implementation
    labels, division_history = diana_clustering(X, n_clusters, metric)
    
    return {
        "labels": labels,
        "division_history": division_history,
        "n_clusters": n_clusters,
    }


def diana_clustering(X: np.ndarray, n_clusters: int, metric: str = "euclidean") -> tuple:
    """
    Custom DIANA implementation using divisive approach.
    
    The algorithm works as follows:
    1. Start with all points in one cluster
    2. Select the cluster with the largest diameter (most scattered)
    3. Split it into two clusters using the splinter method
    4. Repeat until desired number of clusters is reached
    
    Args:
        X: Feature array (n_samples, n_features)
        n_clusters: Desired number of clusters
        metric: Distance metric
        
    Returns:
        Tuple of (labels, division_history)
    """
    n_samples = X.shape[0]
    
    # Initialize: all points in cluster 0
    labels = np.zeros(n_samples, dtype=int)
    division_history = []
    
    # Current number of clusters
    current_clusters = 1
    
    while current_clusters < n_clusters:
        # Find the cluster with maximum diameter (most scattered)
        max_diameter = -1
        cluster_to_split = 0
        
        for cluster_id in range(current_clusters):
            cluster_mask = labels == cluster_id
            cluster_points = X[cluster_mask]
            
            if len(cluster_points) < 2:
                continue
            
            # Calculate diameter as max pairwise distance
            diameter = compute_cluster_diameter(cluster_points, metric)
            
            if diameter > max_diameter:
                max_diameter = diameter
                cluster_to_split = cluster_id
        
        # Get points in the cluster to split
        cluster_mask = labels == cluster_to_split
        cluster_points = X[cluster_mask]
        cluster_indices = np.where(cluster_mask)[0]
        
        if len(cluster_points) < 2:
            break  # Cannot split anymore
        
        # Split using splinter method (true DIANA approach)
        split_labels = splinter_split(cluster_points, metric)
        
        # Update labels
        new_cluster_id = current_clusters
        for idx, local_label in zip(cluster_indices, split_labels):
            if local_label == 1:
                labels[idx] = new_cluster_id
        
        # Record division
        division_history.append({
            "split_cluster": cluster_to_split,
            "new_cluster": new_cluster_id,
            "split_diameter": max_diameter,
        })
        
        current_clusters += 1
    
    return labels, division_history


def compute_cluster_diameter(points: np.ndarray, metric: str = "euclidean") -> float:
    """
    Compute the diameter of a cluster (maximum pairwise distance).
    
    Args:
        points: Array of points in the cluster
        metric: Distance metric
        
    Returns:
        Cluster diameter
    """
    if len(points) < 2:
        return 0.0
    
    dist_matrix = _compute_pairwise_distances(points, metric)
    return float(np.max(dist_matrix))


def splinter_split(points: np.ndarray, metric: str = "euclidean") -> np.ndarray:
    """
    Original DIANA splinter method for binary splitting.
    
    Algorithm:
    1. Find the point with highest average dissimilarity to others (splinter)
    2. Move splinter to new group B, rest stays in group A
    3. Iteratively: for each point in A, if avg distance to B < avg distance to A,
       move point to B
    4. Repeat step 3 until no more movements
    
    Args:
        points: Array of points to split
        metric: Distance metric
        
    Returns:
        Binary labels (0 or 1) for each point
    """
    n_points = len(points)
    
    if n_points < 2:
        return np.zeros(n_points, dtype=int)
    
    # Compute pairwise distance matrix
    dist_matrix = _compute_pairwise_distances(points, metric)
    
    # Find splinter: point with maximum average distance to all others
    avg_distances = dist_matrix.sum(axis=1) / (n_points - 1)
    splinter_idx = np.argmax(avg_distances)
    
    # Initialize groups: A = all points except splinter, B = {splinter}
    group_B = {splinter_idx}
    group_A = set(range(n_points)) - group_B
    
    # Iteratively move points
    changed = True
    while changed:
        changed = False
        points_to_move = []
        
        for i in list(group_A):  # Convert to list to avoid set modification during iteration
            # Average distance to group A (excluding self)
            a_indices = list(group_A - {i})
            if len(a_indices) == 0:
                avg_dist_A = float('inf')
            else:
                avg_dist_A = np.mean([dist_matrix[i, j] for j in a_indices])
            
            # Average distance to group B
            b_indices = list(group_B)
            avg_dist_B = np.mean([dist_matrix[i, j] for j in b_indices])
            
            # Move to B if closer to B
            if avg_dist_B < avg_dist_A:
                points_to_move.append(i)
        
        if points_to_move:
            for i in points_to_move:
                group_A.remove(i)
                group_B.add(i)
            changed = True
    
    # Create labels
    labels = np.zeros(n_points, dtype=int)
    for i in group_B:
        labels[i] = 1
    
    return labels


def compute_diana_analysis(X: np.ndarray, metric: str = "euclidean", 
                           k_range: range = range(2, 11)) -> dict:
    """
    Compute DIANA clustering analysis for multiple cluster numbers.
    
    Args:
        X: Feature array
        metric: Distance metric
        k_range: Range of cluster numbers to test
        
    Returns:
        Dictionary with analysis results
    """
    from sklearn.metrics import silhouette_score
    
    results = {
        "k_values": list(k_range),
        "silhouette_scores": [],
    }
    
    for k in k_range:
        labels, _ = diana_clustering(X, k, metric)
        
        n_unique = len(np.unique(labels))
        if n_unique >= 2:
            score = silhouette_score(X, labels)
        else:
            score = 0
        
        results["silhouette_scores"].append(score)
    
    return results
