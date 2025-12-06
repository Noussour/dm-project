"""
Plotting functions for clustering visualization.
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram

from config.constants import COLOR_PALETTE


def create_2d_scatter(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    labels: np.ndarray,
    title: str,
    centroids: np.ndarray = None,
    feature_names: list = None,
) -> go.Figure:
    """
    Create a 2D scatter plot colored by cluster labels.
    
    Args:
        df: DataFrame with features
        x_col: Column name for X axis
        y_col: Column name for Y axis
        labels: Cluster labels
        title: Plot title
        centroids: Optional centroid coordinates
        feature_names: List of feature names (for centroid mapping)
        
    Returns:
        Plotly Figure object
    """
    plot_df = df.copy()
    plot_df["_label"] = labels
    
    fig = px.scatter(
        plot_df,
        x=x_col,
        y=y_col,
        color=plot_df["_label"].astype(str),
        symbol=plot_df["_label"].astype(str),
        title=title,
        labels={"color": "Cluster"},
        color_discrete_sequence=COLOR_PALETTE,
    )
    
    # Add centroids if available
    if centroids is not None and feature_names is not None:
        try:
            c_df = pd.DataFrame(centroids, columns=feature_names)
            fig.add_trace(go.Scatter(
                x=c_df[x_col],
                y=c_df[y_col],
                mode="markers",
                marker=dict(symbol="x", size=12, line=dict(width=1), color="black"),
                name="Centroids"
            ))
        except Exception:
            pass
    
    return fig


def create_3d_scatter(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    z_col: str,
    labels: np.ndarray,
    title: str,
    centroids: np.ndarray = None,
    feature_names: list = None,
) -> go.Figure:
    """
    Create a 3D scatter plot colored by cluster labels.
    
    Args:
        df: DataFrame with features
        x_col: Column name for X axis
        y_col: Column name for Y axis
        z_col: Column name for Z axis
        labels: Cluster labels
        title: Plot title
        centroids: Optional centroid coordinates
        feature_names: List of feature names (for centroid mapping)
        
    Returns:
        Plotly Figure object
    """
    plot_df = df.copy()
    plot_df["_label"] = labels
    
    fig = px.scatter_3d(
        plot_df,
        x=x_col,
        y=y_col,
        z=z_col,
        color=plot_df["_label"].astype(str),
        symbol=plot_df["_label"].astype(str),
        title=title,
        color_discrete_sequence=COLOR_PALETTE,
        labels={"color": "Cluster"}
    )
    
    # Add centroids if available
    if centroids is not None and feature_names is not None:
        try:
            c_df = pd.DataFrame(centroids, columns=feature_names)
            fig.add_trace(go.Scatter3d(
                x=c_df[x_col],
                y=c_df[y_col],
                z=c_df[z_col],
                mode="markers",
                marker=dict(symbol="x", size=8, line=dict(width=1), color="black"),
                name="Centroids"
            ))
        except Exception:
            pass
    
    return fig


def create_elbow_plot(k_values: list, inertias: list) -> go.Figure:
    """
    Create an elbow plot for K-Means.
    
    Args:
        k_values: List of k values tested
        inertias: List of corresponding inertia values
        
    Returns:
        Plotly Figure object
    """
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=k_values,
        y=inertias,
        mode="lines+markers",
        name="Inertia"
    ))
    fig.update_layout(
        xaxis_title="n_clusters",
        yaxis_title="Inertia",
        title="Elbow Curve (K-Means)"
    )
    return fig


def create_dendrogram(
    linkage_matrix: np.ndarray,
    n_clusters: int = 2,
    figsize: tuple = (10, 4),
) -> plt.Figure:
    """
    Create a dendrogram plot with a horizontal cut line.
    
    Args:
        linkage_matrix: Linkage matrix from scipy.cluster.hierarchy.linkage
        n_clusters: Number of clusters for the cut line
        figsize: Figure size tuple
        
    Returns:
        Matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    dendrogram(
        linkage_matrix,
        ax=ax,
        truncate_mode=None,
        no_labels=True,
        color_threshold=None
    )
    
    # Find threshold height for the cut
    distances = linkage_matrix[:, 2]
    idx = max(0, len(distances) - n_clusters + 1)
    
    try:
        threshold = distances[idx]
    except Exception:
        threshold = distances.mean()
    
    ax.axhline(y=threshold, color="red", linestyle="--", label=f"cut (k={n_clusters})")
    ax.set_title("Dendrogram")
    ax.legend()
    
    return fig


def create_cluster_histogram(labels: np.ndarray) -> go.Figure:
    """
    Create a histogram of cluster sizes.
    
    Args:
        labels: Cluster labels
        
    Returns:
        Plotly Figure object
    """
    unique, counts = np.unique(labels, return_counts=True)
    labels_str = [str(u) for u in unique]
    
    fig = go.Figure()
    fig.add_trace(go.Bar(x=labels_str, y=counts))
    fig.update_layout(
        xaxis_title="Cluster (label)",
        yaxis_title="Number of observations",
        title="Cluster Sizes"
    )
    
    return fig


def create_silhouette_plot(k_values: list, silhouette_scores: list) -> go.Figure:
    """
    Create a silhouette score vs K plot.
    
    Args:
        k_values: List of k values tested
        silhouette_scores: List of corresponding silhouette scores
        
    Returns:
        Plotly Figure object
    """
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=k_values,
        y=silhouette_scores,
        mode="lines+markers",
        name="Silhouette Score",
        line=dict(color="royalblue", width=2),
        marker=dict(size=8)
    ))
    fig.update_layout(
        xaxis_title="Number of clusters (k)",
        yaxis_title="Silhouette Score",
        title="Silhouette Score vs. K",
        hovermode="x unified"
    )
    return fig


def create_kdistance_graph(distances: np.ndarray, k: int = 4) -> go.Figure:
    """
    Create a k-distance graph for DBSCAN epsilon selection.
    
    Args:
        distances: Array of k-distances (sorted)
        k: k value for k-distance (typically k = min_samples)
        
    Returns:
        Plotly Figure object
    """
    sorted_distances = np.sort(distances)[::-1]  # Sort in descending order
    indices = np.arange(len(sorted_distances))
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=indices,
        y=sorted_distances,
        mode="lines",
        name=f"{k}-Distance",
        line=dict(color="darkred", width=1)
    ))
    fig.update_layout(
        xaxis_title="Points (sorted)",
        yaxis_title=f"{k}-Distance",
        title=f"k-Distance Graph (k={k})",
        hovermode="x unified"
    )
    return fig
