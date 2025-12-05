"""
AGNES (Agglomerative Nesting) - Bottom-up hierarchical clustering implementation.

AGNES is a bottom-up approach where each observation starts as its own cluster,
and pairs of clusters are merged as one moves up the hierarchy.

Custom implementation from scratch - no sklearn dependency.
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


def _single_linkage(dist_matrix: np.ndarray, cluster_i: list, cluster_j: list) -> float:
    """Single linkage: minimum distance between clusters."""
    min_dist = np.inf
    for i in cluster_i:
        for j in cluster_j:
            if dist_matrix[i, j] < min_dist:
                min_dist = dist_matrix[i, j]
    return min_dist


def _complete_linkage(dist_matrix: np.ndarray, cluster_i: list, cluster_j: list) -> float:
    """Complete linkage: maximum distance between clusters."""
    max_dist = -np.inf
    for i in cluster_i:
        for j in cluster_j:
            if dist_matrix[i, j] > max_dist:
                max_dist = dist_matrix[i, j]
    return max_dist


def _average_linkage(dist_matrix: np.ndarray, cluster_i: list, cluster_j: list) -> float:
    """Average linkage: mean distance between clusters."""
    total_dist = 0.0
    count = 0
    for i in cluster_i:
        for j in cluster_j:
            total_dist += dist_matrix[i, j]
            count += 1
    return total_dist / count if count > 0 else 0.0


def _ward_linkage(X: np.ndarray, cluster_i: list, cluster_j: list) -> float:
    """
    Ward linkage: minimize within-cluster variance.
    Uses the Lance-Williams formula for Ward's method.
    """
    points_i = X[cluster_i]
    points_j = X[cluster_j]
    
    centroid_i = np.mean(points_i, axis=0)
    centroid_j = np.mean(points_j, axis=0)
    
    ni = len(cluster_i)
    nj = len(cluster_j)
    
    # Ward distance formula
    dist = np.sqrt((2 * ni * nj) / (ni + nj)) * _euclidean_distance(centroid_i, centroid_j)
    return dist


def agnes_custom(X: np.ndarray, n_clusters: int = 2, linkage: str = "ward", 
                 metric: str = "euclidean") -> tuple:
    """
    Custom AGNES implementation from scratch.
    
    Args:
        X: Data array of shape (n_samples, n_features)
        n_clusters: Number of clusters to form
        linkage: Linkage method ('single', 'complete', 'average', 'ward')
        metric: Distance metric ('euclidean', 'manhattan')
        
    Returns:
        Tuple of (labels, linkage_matrix)
    """
    n_samples = X.shape[0]
    
    # Initialize each point as its own cluster
    clusters = {i: [i] for i in range(n_samples)}
    
    # Compute initial distance matrix
    dist_matrix = _compute_distance_matrix(X, metric)
    
    # Linkage matrix for dendrogram: [idx1, idx2, distance, n_points]
    linkage_matrix = []
    
    # Next cluster index (for merged clusters)
    next_cluster_idx = n_samples
    
    # Map original indices to current cluster indices
    cluster_map = {i: i for i in range(n_samples)}
    
    # Merge until we have desired number of clusters
    while len(clusters) > n_clusters:
        # Find the two closest clusters
        min_dist = np.inf
        merge_i, merge_j = None, None
        
        cluster_keys = list(clusters.keys())
        
        for idx_i, key_i in enumerate(cluster_keys):
            for idx_j in range(idx_i + 1, len(cluster_keys)):
                key_j = cluster_keys[idx_j]
                
                # Compute distance based on linkage method
                if linkage == "ward":
                    dist = _ward_linkage(X, clusters[key_i], clusters[key_j])
                elif linkage == "single":
                    dist = _single_linkage(dist_matrix, clusters[key_i], clusters[key_j])
                elif linkage == "complete":
                    dist = _complete_linkage(dist_matrix, clusters[key_i], clusters[key_j])
                else:  # average
                    dist = _average_linkage(dist_matrix, clusters[key_i], clusters[key_j])
                
                if dist < min_dist:
                    min_dist = dist
                    merge_i, merge_j = key_i, key_j
        
        # Merge the two closest clusters
        new_cluster = clusters[merge_i] + clusters[merge_j]
        n_points = len(new_cluster)
        
        # Record in linkage matrix
        linkage_matrix.append([merge_i, merge_j, min_dist, n_points])
        
        # Update clusters
        del clusters[merge_i]
        del clusters[merge_j]
        clusters[next_cluster_idx] = new_cluster
        
        next_cluster_idx += 1
    
    # Assign labels
    labels = np.zeros(n_samples, dtype=int)
    for cluster_idx, (_, points) in enumerate(clusters.items()):
        for point_idx in points:
            labels[point_idx] = cluster_idx
    
    return labels, np.array(linkage_matrix)


def run_agnes(X: np.ndarray, params: dict) -> dict:
    """
    Execute AGNES (Agglomerative Nesting) hierarchical clustering.
    
    Args:
        X: Feature array
        params: Dictionary with parameters:
            - 'n_clusters': Number of clusters (default: 2)
            - 'linkage': Linkage method ('ward', 'complete', 'average', 'single') (default: 'ward')
            - 'metric': Distance metric (default: 'euclidean')
              Note: 'ward' linkage requires 'euclidean' metric
        
    Returns:
        Dictionary with labels and linkage matrix
    """
    n_clusters = int(params.get("n_clusters", 2))
    linkage_method = params.get("linkage", "ward")
    metric = params.get("metric", "euclidean")
    
    # Ward linkage only works with euclidean distance
    if linkage_method == "ward":
        metric = "euclidean"
    
    # Use custom implementation
    labels, linkage_matrix = agnes_custom(X, n_clusters, linkage_method, metric)
    
    # Compute full linkage matrix for dendrogram (always computed to k=1)
    try:
        full_linkage_matrix = compute_linkage_matrix(X, linkage_method, metric)
    except Exception:
        full_linkage_matrix = linkage_matrix
    
    return {
        "labels": labels,
        "linkage_matrix": full_linkage_matrix,
        "n_clusters": n_clusters,
    }


def compute_linkage_matrix(X: np.ndarray, method: str = "ward", 
                           metric: str = "euclidean") -> np.ndarray:
    """
    Compute full linkage matrix for dendrogram visualization.
    This computes the complete hierarchical structure (until 1 cluster).
    
    Args:
        X: Feature array
        method: Linkage method ('ward', 'complete', 'average', 'single')
        metric: Distance metric
        
    Returns:
        Linkage matrix Z compatible with scipy.cluster.hierarchy.dendrogram
    """
    n_samples = X.shape[0]
    
    # Initialize each point as its own cluster
    clusters = {i: [i] for i in range(n_samples)}
    
    # Compute initial distance matrix
    dist_matrix = _compute_distance_matrix(X, metric)
    
    # Linkage matrix for dendrogram: [idx1, idx2, distance, n_points]
    linkage_matrix = []
    
    # Next cluster index (for merged clusters)
    next_cluster_idx = n_samples
    
    # Merge until we have 1 cluster
    while len(clusters) > 1:
        # Find the two closest clusters
        min_dist = np.inf
        merge_i, merge_j = None, None
        
        cluster_keys = list(clusters.keys())
        
        for idx_i, key_i in enumerate(cluster_keys):
            for idx_j in range(idx_i + 1, len(cluster_keys)):
                key_j = cluster_keys[idx_j]
                
                # Compute distance based on linkage method
                if method == "ward":
                    dist = _ward_linkage(X, clusters[key_i], clusters[key_j])
                elif method == "single":
                    dist = _single_linkage(dist_matrix, clusters[key_i], clusters[key_j])
                elif method == "complete":
                    dist = _complete_linkage(dist_matrix, clusters[key_i], clusters[key_j])
                else:  # average
                    dist = _average_linkage(dist_matrix, clusters[key_i], clusters[key_j])
                
                if dist < min_dist:
                    min_dist = dist
                    merge_i, merge_j = key_i, key_j
        
        # Merge the two closest clusters
        new_cluster = clusters[merge_i] + clusters[merge_j]
        n_points = len(new_cluster)
        
        # Record in linkage matrix
        linkage_matrix.append([merge_i, merge_j, min_dist, n_points])
        
        # Update clusters
        del clusters[merge_i]
        del clusters[merge_j]
        clusters[next_cluster_idx] = new_cluster
        
        next_cluster_idx += 1
    
    return np.array(linkage_matrix)


# Alias for backwards compatibility
def compute_agnes_linkage(X: np.ndarray, method: str = "ward") -> np.ndarray:
    """Alias for compute_linkage_matrix."""
    return compute_linkage_matrix(X, method)


def get_clusters_from_linkage(linkage_matrix: np.ndarray, n_samples: int, 
                               n_clusters: int) -> np.ndarray:
    """
    Get cluster assignments by cutting the dendrogram to get n_clusters.
    
    Args:
        linkage_matrix: Linkage matrix from AGNES
        n_samples: Number of original samples
        n_clusters: Number of clusters desired
        
    Returns:
        Array of cluster labels
    """
    # Start with each point in its own cluster
    labels = np.arange(n_samples)
    
    # Number of merges to perform = n_samples - n_clusters
    n_merges = n_samples - n_clusters
    
    # Apply merges
    next_cluster_idx = n_samples
    cluster_map = {i: i for i in range(n_samples)}
    
    for i in range(n_merges):
        idx1, idx2 = int(linkage_matrix[i, 0]), int(linkage_matrix[i, 1])
        
        # Find all points in cluster idx2 and reassign to idx1
        label1 = cluster_map.get(idx1, idx1)
        label2 = cluster_map.get(idx2, idx2)
        
        # Merge: change all label2 to label1
        labels[labels == label2] = label1
        
        # Update cluster map
        cluster_map[next_cluster_idx] = label1
        next_cluster_idx += 1
    
    # Relabel to 0, 1, 2, ...
    unique_labels = np.unique(labels)
    label_map = {old: new for new, old in enumerate(unique_labels)}
    labels = np.array([label_map[l] for l in labels])
    
    return labels


def get_agnes_clusters_at_distance(X: np.ndarray, linkage_matrix: np.ndarray, 
                                    distance_threshold: float) -> np.ndarray:
    """
    Get cluster assignments by cutting the dendrogram at a specific distance.
    
    Args:
        X: Feature array
        linkage_matrix: Precomputed linkage matrix
        distance_threshold: Distance at which to cut the dendrogram
        
    Returns:
        Array of cluster labels
    """
    n_samples = X.shape[0]
    
    # Start with each point in its own cluster
    labels = np.arange(n_samples)
    
    next_cluster_idx = n_samples
    cluster_map = {i: i for i in range(n_samples)}
    
    for i in range(len(linkage_matrix)):
        if linkage_matrix[i, 2] > distance_threshold:
            break
        
        idx1, idx2 = int(linkage_matrix[i, 0]), int(linkage_matrix[i, 1])
        
        # Find all points in cluster idx2 and reassign to idx1
        label1 = cluster_map.get(idx1, idx1)
        label2 = cluster_map.get(idx2, idx2)
        
        # Merge: change all label2 to label1
        labels[labels == label2] = label1
        
        # Update cluster map
        cluster_map[next_cluster_idx] = label1
        next_cluster_idx += 1
    
    # Relabel to 0, 1, 2, ...
    unique_labels = np.unique(labels)
    label_map = {old: new for new, old in enumerate(unique_labels)}
    labels = np.array([label_map[l] for l in labels])
    
    return labels


def compute_agnes_analysis(X: np.ndarray, method: str = "ward", 
                           k_range: range = range(2, 11),
                           metric: str = "euclidean") -> dict:
    """
    Compute AGNES clustering analysis for multiple cluster numbers.
    
    Args:
        X: Feature array
        method: Linkage method
        k_range: Range of cluster numbers to test
        metric: Distance metric
        
    Returns:
        Dictionary with analysis results
    """
    from sklearn.metrics import silhouette_score
    
    # Compute full linkage matrix once
    linkage_matrix = compute_linkage_matrix(X, method, metric)
    n_samples = X.shape[0]
    
    results = {
        "k_values": list(k_range),
        "silhouette_scores": [],
        "linkage_matrix": linkage_matrix,
    }
    
    for k in k_range:
        labels = get_clusters_from_linkage(linkage_matrix, n_samples, k)
        
        if len(np.unique(labels)) >= 2:
            try:
                score = silhouette_score(X, labels)
            except Exception:
                score = 0
        else:
            score = 0
        
        results["silhouette_scores"].append(score)
    
    return results
