"""
Tab components for main content area.
"""

import numpy as np
import pandas as pd
import streamlit as st

from clustering import (
    run_clustering, 
    compute_elbow, 
    compute_silhouette_scores,
    compute_linkage_matrix,
    compute_kdistances,
)
from visualization import (
    create_2d_scatter,
    create_3d_scatter,
    create_elbow_plot,
    create_dendrogram,
    create_cluster_histogram,
    create_silhouette_plot,
    create_kdistance_graph,
)
from utils.metrics import n_clusters_from_labels
from config.constants import MAX_3D_POINTS


def get_results_list() -> list:
    """Get list of stored results in chronological order."""
    items = st.session_state.get("results", {})
    return list(items.items())


def get_compatible_runs(df: pd.DataFrame) -> list[str]:
    """
    Get list of run keys that are compatible with the current DataFrame.
    A run is compatible if all its features exist in the current DataFrame.
    
    Args:
        df: Current DataFrame
        
    Returns:
        List of compatible run keys
    """
    compatible = []
    current_cols = set(df.columns.tolist())
    
    for key, val in st.session_state.get("results", {}).items():
        features_used = set(val.get("features", []))
        if features_used.issubset(current_cols):
            compatible.append(key)
    
    return compatible


def render_visualization_tab(df: pd.DataFrame, selected_features: list[str]):
    """
    Render the 2D/3D visualization tab.
    
    Args:
        df: Full DataFrame
        selected_features: List of selected feature names
    """
    from utils.data_loader import validate_visualization_compatibility
    
    st.subheader("Interactive Visualization")
    
    # Validate minimum features for visualization
    if len(selected_features) < 2:
        st.error("❌ Visualization requires at least 2 selected features.")
        st.info("💡 Return to preprocessing and select at least 2 numeric features.")
        return
    
    st.markdown(
        "Choose X / Y (optional Z) axes from selected numeric features."
    )
    
    # Axis selection
    col1, col2, col3 = st.columns(3)
    
    with col1:
        x_axis = st.selectbox("X Axis", selected_features, index=0)
    with col2:
        y_axis = st.selectbox("Y Axis", selected_features, index=1 if len(selected_features) > 1 else 0)
    with col3:
        # Z axis with option to disable 3D (None = 2D only)
        if len(selected_features) >= 3:
            z_options = [None] + selected_features
            z_default_index = 0  # Default to "Disable 3D"
            z_axis = st.selectbox(
                "Z Axis (3D)",
                z_options,
                index=z_default_index,
                format_func=lambda x: "Disable 3D" if x is None else x
            )
        else:
            z_axis = None
            st.info("📊 3D disabled (requires 3+ features)")
    
    z_option = z_axis is not None
    
    # Validate 3D visualization if selected
    if z_option:
        is_valid, error_msg = validate_visualization_compatibility(
            df, selected_features, "", "3D_scatter"
        )
        if not is_valid:
            st.warning(f"⚠️ {error_msg}")
            z_option = False
            z_axis = None
    
    # Check for stored results
    runs = get_results_list()
    
    if len(runs) == 0:
        st.info("ℹ️ No stored clustering run. Launch an algorithm via the sidebar.")
        return
    
    # Filter runs compatible with current dataset
    compatible_runs = get_compatible_runs(df)
    
    if len(compatible_runs) == 0:
        st.warning(
            "No run compatible with current dataset. "
            "Previous runs used different features. "
            "Execute a new clustering on this dataset."
        )
        return
    
    # Run selection (only compatible runs)
    sel = st.selectbox(
        "Choose a run to display",
        options=compatible_runs,
        format_func=lambda k: f"{k} — {st.session_state['results'][k]['result']['algo']} ({st.session_state['results'][k]['dataset']})"
    )
    
    chosen = st.session_state["results"][sel]
    res = chosen["result"]
    features_used = chosen["features"]
    
    # Prepare plot data
    vis_sample = df[selected_features]
    if vis_sample.shape[0] > MAX_3D_POINTS:
        st.info(f"For visualization, sampling to {MAX_3D_POINTS} points.")
        plot_df = vis_sample.sample(MAX_3D_POINTS, random_state=42)
    else:
        plot_df = vis_sample.copy()
    
    # Recompute labels for sampled data
    try:
        labels_plot = run_clustering(res["algo"], res["params"], plot_df)["labels"]
    except Exception:
        labels_plot = res["labels"]
    
    centroids = res.get("centroids", None)
    plot_df = plot_df.reset_index(drop=True)
    
    # Create and display plot
    if z_option and z_axis is not None:
        fig = create_3d_scatter(
            plot_df, x_axis, y_axis, z_axis,
            labels_plot,
            f"{res['algo']} — 3D Visualization ({sel})",
            centroids, features_used
        )
    else:
        fig = create_2d_scatter(
            plot_df, x_axis, y_axis,
            labels_plot,
            f"{res['algo']} — 2D Visualization ({sel})",
            centroids, features_used
        )
    
    st.plotly_chart(fig, width='stretch')
    
    # Legend and help
    st.markdown(
        "**Legend** : each symbol/color represents a cluster. "
        "Gray often indicates noise (DBSCAN)."
    )
    
    with st.expander("Quick Metric Definitions (click to open)"):
        st.markdown("""
**Silhouette Score** — measure of cohesion / separation. Value in [-1,1], higher = better separation.

**Calinski-Harabasz** — ratio of inter / intra-cluster variance (higher = better).

**Davies-Bouldin** — average intra- / inter-cluster similarity (lower = better).

**Inertia (WCSS)** — sum of intra-cluster squared distances (lower = more compact clusters).
""")


def render_metrics_tab():
    """Render the metrics comparison tab."""
    st.subheader("Quantitative comparison of stored runs")
    
    runs = get_results_list()
    
    if len(runs) == 0:
        st.info("No stored results. Run at least one algorithm to compare.")
        return
    
    # Build summary DataFrame for metrics
    metrics_rows = []
    params_rows = []
    
    for key, val in st.session_state["results"].items():
        r = val["result"]
        metrics = r.get("metrics", {})
        params = r.get("params", {})
        
        # Metrics row
        metrics_rows.append({
            "Run ID": key,
            "Algorithme": r["algo"],
            "Dataset": val["dataset"],
            "Features": ", ".join(val["features"]),
            "N° Clusters": n_clusters_from_labels(r.get("labels", np.array([]))),
            "Silhouette ↑": round(metrics.get("silhouette"), 4) if metrics.get("silhouette") else None,
            "Calinski-Harabasz ↑": round(metrics.get("calinski_harabasz"), 2) if metrics.get("calinski_harabasz") else None,
            "Davies-Bouldin ↓": round(metrics.get("davies_bouldin"), 4) if metrics.get("davies_bouldin") else None,
            "Inertie (WCSS) ↓": round(metrics.get("inertia"), 2) if metrics.get("inertia") else None,
        })
        
        # Parameters row
        params_row = {
            "Run ID": key,
            "Algorithme": r["algo"],
        }
        # Add all parameters dynamically
        for param_name, param_value in params.items():
            params_row[param_name] = param_value
        params_rows.append(params_row)
    
    metrics_df = pd.DataFrame(metrics_rows)
    params_df = pd.DataFrame(params_rows)
    
    # Display metrics table
    st.markdown("### Evaluation Metrics Table")
    st.markdown("*↑ = higher = better, ↓ = lower = better*")
    st.dataframe(metrics_df, width='stretch', hide_index=True)
    
    # Display parameters table
    st.markdown("### Used Parameters Table")
    st.dataframe(params_df, width='stretch', hide_index=True)
    
    # Side-by-side comparison
    st.markdown("---")
    st.markdown("### Side-by-side Comparison")
    colA, colB = st.columns(2)
    keys = list(st.session_state["results"].keys())
    
    with colA:
        selA = st.selectbox(
            "Run A", keys, index=0,
            format_func=lambda k: f"{k} — {st.session_state['results'][k]['result']['algo']}"
        )
    with colB:
        selB = st.selectbox(
            "Run B", keys, index=min(1, len(keys) - 1),
            format_func=lambda k: f"{k} — {st.session_state['results'][k]['result']['algo']}"
        )
    
    rA = st.session_state["results"][selA]["result"]
    rB = st.session_state["results"][selB]["result"]
    mA = rA.get("metrics", {})
    mB = rB.get("metrics", {})
    pA = rA.get("params", {})
    pB = rB.get("params", {})
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Metrics**")
        comp_metrics_df = pd.DataFrame({
            "Metric": ["Effective Clusters", "Silhouette ↑", "Calinski-Harabasz ↑", "Davies-Bouldin ↓", "Inertia (WCSS) ↓"],
            f"Run A ({rA['algo']})": [
                n_clusters_from_labels(rA.get("labels", np.array([]))),
                round(mA.get("silhouette"), 4) if mA.get("silhouette") else "N/A",
                round(mA.get("calinski_harabasz"), 2) if mA.get("calinski_harabasz") else "N/A",
                round(mA.get("davies_bouldin"), 4) if mA.get("davies_bouldin") else "N/A",
                round(mA.get("inertia"), 2) if mA.get("inertia") else "N/A",
            ],
            f"Run B ({rB['algo']})": [
                n_clusters_from_labels(rB.get("labels", np.array([]))),
                round(mB.get("silhouette"), 4) if mB.get("silhouette") else "N/A",
                round(mB.get("calinski_harabasz"), 2) if mB.get("calinski_harabasz") else "N/A",
                round(mB.get("davies_bouldin"), 4) if mB.get("davies_bouldin") else "N/A",
                round(mB.get("inertia"), 2) if mB.get("inertia") else "N/A",
            ]
        })
        st.table(comp_metrics_df)
    
    with col2:
        st.markdown("**Parameters**")
        # Collect all unique parameter names
        all_params = set(pA.keys()) | set(pB.keys())
        param_rows = []
        for param in sorted(all_params):
            param_rows.append({
                "Parameter": param,
                f"Run A ({rA['algo']})": pA.get(param, "N/A"),
                f"Run B ({rB['algo']})": pB.get(param, "N/A"),
            })
        if param_rows:
            comp_params_df = pd.DataFrame(param_rows)
            st.table(comp_params_df)
        else:
            st.info("No parameters to compare.")


def render_charts_tab(df: pd.DataFrame, selected_features: list[str]):
    """
    Render the additional charts tab with algorithm-specific visualizations.
    
    Args:
        df: Full DataFrame
        selected_features: List of selected feature names
    """
    from utils.data_loader import validate_visualization_compatibility
    from config.constants import VISUALIZATION_CONSTRAINTS
    
    st.subheader("Additional Charts")
    
    runs = get_results_list()
    
    if len(runs) == 0:
        st.info("ℹ️ Run algorithms to generate these charts.")
        return
    
    # Filter runs compatible with current dataset
    compatible_runs = get_compatible_runs(df)
    
    if len(compatible_runs) == 0:
        st.warning(
            "⚠️ No run compatible with current dataset. "
            "Previous runs used different features. "
            "Execute a new clustering on this dataset."
        )
        return
    
    # Run selection (only compatible runs)
    sel_run = st.selectbox(
        "Choose a run for charts",
        options=compatible_runs,
        format_func=lambda k: f"{k} — {st.session_state['results'][k]['result']['algo']}"
    )
    
    chosen = st.session_state["results"][sel_run]
    res = chosen["result"]
    features_used = chosen["features"]
    algo = res["algo"]
    
    # Validate features exist in current dataframe
    missing_features = [f for f in features_used if f not in df.columns]
    missing_features = [f for f in features_used if f not in df.columns]
    if missing_features:
        st.error(f"❌ Missing features in current dataset: {missing_features}")
        return
    
    X = df[features_used]
    X_np = X.to_numpy()
    
    # Show algorithm-specific chart availability info
    st.markdown("---")
    st.markdown(f"**Selected Algorithm**: `{algo}`")
    
    # Display which visualizations are available for this algorithm
    available_viz = []
    if algo in ["KMeans", "K-Medoids"]:
        available_viz = ["Elbow Plot", "Silhouette vs K", "Cluster Distribution"]
    elif algo == "DBSCAN":
        available_viz = ["k-Distance Graph", "Cluster Distribution"]
    elif algo in ["AGNES", "DIANA"]:
        available_viz = ["Dendrogram", "Silhouette vs K", "Cluster Distribution"]
        # Check dendrogram size limit
        if len(df) > VISUALIZATION_CONSTRAINTS["dendrogram"]["max_samples"]:
            st.warning(f"⚠️ Dendrogram disabled as dataset contains more than "
                      f"{VISUALIZATION_CONSTRAINTS['dendrogram']['max_samples']} samples ({len(df)}).")
    
    st.info(f"📊 Available visualizations for {algo}: {', '.join(available_viz)}")

    # Dispatch charts based on algorithm
    if algo == "KMeans":
        _render_kmeans_charts(X_np, res, sel_run)
    elif algo == "K-Medoids":
        _render_kmedoids_charts(X_np, res, sel_run)
    elif algo == "DBSCAN":
        _render_dbscan_charts(X_np, res, sel_run)
    elif algo == "AGNES":
        _render_agnes_charts(X_np, res, sel_run, len(df))
    elif algo == "DIANA":
        _render_diana_charts(X_np, res, sel_run, len(df))
    else:
        # Generic chart for unknown algorithms
        _render_generic_charts(X_np, res, sel_run)


    


def _render_diana_charts(X: np.ndarray, res: dict, run_name: str, n_samples: int = None):
    """Render DIANA specific charts: Dendrogram + Silhouette + Histogram"""
    from config.constants import VISUALIZATION_CONSTRAINTS
    
    n_samples = n_samples or X.shape[0]
    max_dendrogram_samples = VISUALIZATION_CONSTRAINTS["dendrogram"]["max_samples"]
    
    tabs = st.tabs(["Dendrogram", "Silhouette vs K", "Cluster Distribution"])
    
    # Dendrogram
    with tabs[0]:
        if n_samples > max_dendrogram_samples:
            st.warning(f"⚠️ Dendrogram disabled: dataset ({n_samples} samples) "
                      f"exceeds limit of {max_dendrogram_samples} samples.")
            st.info("💡 Use subsampling or 2D/3D visualization instead.")
        else:
            st.markdown("**Dendrogram (DIANA - Divisive)** : hierarchical clustering display.")
            n_clusters = int(res.get("params", {}).get("n_clusters", 2))
            k_cut = st.slider("Cut to obtain how many clusters?", 2, min(10, X.shape[0]), n_clusters, key=f"diana_dendrogram_{run_name}")
            try:
                Z = compute_linkage_matrix(X, method="complete")
                fig = create_dendrogram(Z, k_cut, figsize=(12, 5))
                st.pyplot(fig)
            except Exception as e:
                st.error(f"❌ Unable to compute dendrogram: {e}")
    
    # Silhouette plot
    with tabs[1]:
        st.markdown("**Silhouette Score** : quality measure vs number of clusters.")
        if st.button("Calculate Silhouette", key=f"diana_silhouette_{run_name}"):
            from clustering.diana import compute_diana_analysis
            metric = res.get("params", {}).get("metric", "euclidean")
            analysis = compute_diana_analysis(X, metric=metric, k_range=range(2, min(11, X.shape[0])))
            fig = create_silhouette_plot(analysis["k_values"], analysis["silhouette_scores"])
            fig.update_layout(title="Silhouette Score vs. K (DIANA)")
            st.plotly_chart(fig, use_container_width=True)
    
    # Histogram
    with tabs[2]:
        st.markdown("**Cluster Distribution** : number of points per cluster.")
        labels = res.get("labels", np.array([]))
        fig = create_cluster_histogram(labels)
        st.plotly_chart(fig, use_container_width=True)


def _render_agnes_charts(X: np.ndarray, res: dict, run_name: str, n_samples: int = None):
    """Render AGNES specific charts: Dendrogram + Silhouette + Histogram"""
    from config.constants import VISUALIZATION_CONSTRAINTS
    
    n_samples = n_samples or X.shape[0]
    max_dendrogram_samples = VISUALIZATION_CONSTRAINTS["dendrogram"]["max_samples"]
    
    tabs = st.tabs(["Dendrogram", "Silhouette vs K", "Cluster Distribution"])
    
    # Dendrogram
    with tabs[0]:
        if n_samples > max_dendrogram_samples:
            st.warning(f"⚠️ Dendrogram disabled: dataset ({n_samples} samples) "
                      f"exceeds limit of {max_dendrogram_samples} samples.")
            st.info("💡 Use subsampling or 2D/3D visualization instead.")
        else:
            st.markdown("**Dendrogram (AGNES - Agglomerative)** : hierarchical clustering display.")
            n_clusters = int(res.get("params", {}).get("n_clusters", 2))
            linkage_method = res.get("params", {}).get("linkage", "ward")
            k_cut = st.slider("Cut to obtain how many clusters?", 2, min(10, X.shape[0]), n_clusters, key=f"agnes_dendrogram_{run_name}")
            try:
                Z = compute_linkage_matrix(X, method=linkage_method)
                fig = create_dendrogram(Z, k_cut, figsize=(12, 5))
                st.pyplot(fig)
            except Exception as e:
                st.error(f"❌ Unable to compute dendrogram: {e}")
    
    # Silhouette plot
    with tabs[1]:
        st.markdown("**Silhouette Score** : quality measure vs number of clusters.")
        if st.button("Calculate Silhouette", key=f"agnes_silhouette_{run_name}"):
            from clustering.agnes import compute_agnes_analysis
            linkage_method = res.get("params", {}).get("linkage", "ward")
            analysis = compute_agnes_analysis(X, method=linkage_method, k_range=range(2, min(11, X.shape[0])))
            fig = create_silhouette_plot(analysis["k_values"], analysis["silhouette_scores"])
            fig.update_layout(title="Silhouette Score vs. K (AGNES)")
            st.plotly_chart(fig, use_container_width=True)
    
    # Histogram
    with tabs[2]:
        st.markdown("**Cluster Distribution** : number of points per cluster.")
        labels = res.get("labels", np.array([]))
        fig = create_cluster_histogram(labels)
        st.plotly_chart(fig, use_container_width=True)


def _render_kmeans_charts(X: np.ndarray, res: dict, run_name: str):
    """Render K-Means specific charts: Elbow + Silhouette + Histogram"""
    tabs = st.tabs(["Elbow Plot", "Silhouette vs K", "Cluster Distribution"])

    # Elbow plot
    with tabs[0]:
        st.markdown("**Elbow Curve** : inertia vs number of clusters.")
        if st.button("Calculate Elbow", key=f"kmeans_elbow_{run_name}"):
            k_values, inertias = compute_elbow(X, range(2, min(11, X.shape[0])))
            fig = create_elbow_plot(k_values, inertias)
            st.plotly_chart(fig, use_container_width=True)

    # Silhouette plot
    with tabs[1]:
        st.markdown("**Silhouette Score** : quality measure vs number of clusters.")
        if st.button("Calculate Silhouette", key=f"kmeans_silhouette_{run_name}"):
            k_values, scores = compute_silhouette_scores(X, range(2, min(11, X.shape[0])))
            fig = create_silhouette_plot(k_values, scores)
            st.plotly_chart(fig, use_container_width=True)

    # Histogram
    with tabs[2]:
        st.markdown("**Cluster Distribution** : number of points per cluster.")
        labels = res.get("labels", np.array([]))
        fig = create_cluster_histogram(labels)
        st.plotly_chart(fig, use_container_width=True)


def _render_kmedoids_charts(X: np.ndarray, res: dict, run_name: str):
    """Render K-Medoids specific charts: Elbow + Silhouette + Histogram"""
    tabs = st.tabs(["Elbow Plot", "Silhouette vs K", "Cluster Distribution"])

    # Elbow plot for K-Medoids
    with tabs[0]:
        st.markdown("**Elbow Curve (K-Medoids)** : inertia vs number of clusters.")
        metric = res.get("params", {}).get("metric", "euclidean")
        if st.button("Calculate Elbow", key=f"kmedoids_elbow_{run_name}"):
            from clustering.kmedoids import compute_kmedoids_elbow
            k_values, inertias = compute_kmedoids_elbow(X, range(2, min(11, X.shape[0])), metric=metric)
            fig = create_elbow_plot(k_values, inertias)
            fig.update_layout(title="Elbow Curve (K-Medoids)")
            st.plotly_chart(fig, use_container_width=True)

    # Silhouette plot
    with tabs[1]:
        st.markdown("**Silhouette Score** : quality measure vs number of clusters.")
        if st.button("Calculate Silhouette", key=f"kmedoids_silhouette_{run_name}"):
            k_values, scores = compute_silhouette_scores(X, range(2, min(11, X.shape[0])))
            fig = create_silhouette_plot(k_values, scores)
            st.plotly_chart(fig, use_container_width=True)

    # Histogram
    with tabs[2]:
        st.markdown("**Cluster Distribution** : number of points per cluster.")
        labels = res.get("labels", np.array([]))
        fig = create_cluster_histogram(labels)
        st.plotly_chart(fig, use_container_width=True)


def _render_dbscan_charts(X: np.ndarray, res: dict, run_name: str):
    """Render DBSCAN specific charts: k-Distance + Histogram"""
    tabs = st.tabs(["k-Distance Graph", "Cluster Distribution"])

    # k-Distance graph
    with tabs[0]:
        st.markdown("**k-Distance Graph** : to determine epsilon parameter.")
        min_samples = int(res.get("params", {}).get("min_samples", 5))
        if st.button("Calculate k-Distance", key=f"dbscan_kdist_{run_name}"):
            distances = compute_kdistances(X, k=min_samples)
            fig = create_kdistance_graph(distances, k=min_samples)
            st.plotly_chart(fig, use_container_width=True)

    # Histogram
    with tabs[1]:
        st.markdown("**Cluster Distribution** : number of points per cluster (including noise).")
        labels = res.get("labels", np.array([]))
        fig = create_cluster_histogram(labels)
        st.plotly_chart(fig, use_container_width=True)


def _render_generic_charts(X: np.ndarray, res: dict, run_name: str):
    """Render generic charts for any algorithm: Histogram"""
    st.markdown("**Cluster Distribution** : number of points per cluster.")
    labels = res.get("labels", np.array([]))
    fig = create_cluster_histogram(labels)
    st.plotly_chart(fig, use_container_width=True)

