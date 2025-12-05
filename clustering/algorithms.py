"""
Main clustering orchestrator.
"""

from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd

from .kmeans import run_kmeans
from .kmedoids import run_kmedoids
from .dbscan import run_dbscan
from .agnes import run_agnes
from .diana import run_diana
from utils.metrics import compute_metrics, n_clusters_from_labels


@dataclass
class ClusteringResult:
    """Data class to hold clustering results."""
    labels: np.ndarray
    centroids: np.ndarray | None
    params: dict
    timestamp: str
    algo: str
    metrics: dict
    cluster_sizes: dict
    inertia: float | None = None


def run_clustering(algo_name: str, params: dict, X: pd.DataFrame) -> dict:
    """
    Execute the specified clustering algorithm on the data.
    
    Args:
        algo_name: Name of the algorithm ('KMeans', 'K-Medoids', 'DBSCAN', 'AGNES', 'DIANA')
        params: Dictionary of algorithm parameters
        X: DataFrame with numeric features
        
    Returns:
        Dictionary containing labels, centroids, metrics, and other results
    """
    result = {
        "labels": None,
        "centroids": None,
        "params": params,
        "timestamp": datetime.utcnow().isoformat(),
        "algo": algo_name,
        "inertia": None,
    }
    
    X_np = X.to_numpy()
    
    # Route to appropriate algorithm
    if algo_name == "KMeans":
        algo_result = run_kmeans(X_np, params)
        result["labels"] = algo_result["labels"]
        result["centroids"] = algo_result["centroids"]
        result["inertia"] = algo_result["inertia"]
        
    elif algo_name == "K-Medoids":
        algo_result = run_kmedoids(X_np, params)
        result["labels"] = algo_result["labels"]
        result["centroids"] = algo_result["medoids"]  # Use medoids as centroids
        result["medoid_indices"] = algo_result.get("medoid_indices")
        result["inertia"] = algo_result.get("inertia")
        
    elif algo_name == "DBSCAN":
        algo_result = run_dbscan(X_np, params)
        result["labels"] = algo_result["labels"]
        
    elif algo_name == "AGNES":
        algo_result = run_agnes(X_np, params)
        result["labels"] = algo_result["labels"]
        result["linkage_matrix"] = algo_result.get("linkage_matrix")
        
    elif algo_name == "DIANA":
        algo_result = run_diana(X_np, params)
        result["labels"] = algo_result["labels"]
        result["division_history"] = algo_result.get("division_history")
        
    else:
        raise ValueError(f"Algorithme non supporté: {algo_name}")
    
    # Compute metrics
    result["metrics"] = compute_metrics(X_np, result["labels"])
    
    # Compute cluster sizes
    labels = result["labels"]
    unique, counts = np.unique(labels, return_counts=True)
    result["cluster_sizes"] = dict(zip(map(str, unique.tolist()), counts.tolist()))
    
    return result
