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
    st.subheader("Visualisation interactive")
    st.markdown(
        "Choisissez les axes X / Y / (optionnel Z) parmi les features numériques sélectionnées."
    )
    
    # Axis selection
    col1, col2, col3 = st.columns(3)
    
    with col1:
        x_axis = st.selectbox("Axe X", selected_features, index=0)
    with col2:
        y_axis = st.selectbox("Axe Y", selected_features, index=1 if len(selected_features) > 1 else 0)
    with col3:
        # Z axis with option to disable 3D (None = 2D only)
        z_options = [None] + selected_features
        z_default_index = 0  # Default to "Désactiver 3D"
        z_axis = st.selectbox(
            "Axe Z (3D)",
            z_options,
            index=z_default_index,
            format_func=lambda x: "Désactiver 3D" if x is None else x
        )
    
    z_option = z_axis is not None
    
    # Check for stored results
    runs = get_results_list()
    
    if len(runs) == 0:
        st.info("Aucune exécution de clustering stockée. Lancez un algorithme via la sidebar.")
        return
    
    # Filter runs compatible with current dataset
    compatible_runs = get_compatible_runs(df)
    
    if len(compatible_runs) == 0:
        st.warning(
            "Aucun run compatible avec le dataset actuel. "
            "Les runs précédents utilisaient des features différentes. "
            "Exécutez un nouveau clustering sur ce dataset."
        )
        return
    
    # Run selection (only compatible runs)
    sel = st.selectbox(
        "Choisir un run à afficher",
        options=compatible_runs,
        format_func=lambda k: f"{k} — {st.session_state['results'][k]['result']['algo']} ({st.session_state['results'][k]['dataset']})"
    )
    
    chosen = st.session_state["results"][sel]
    res = chosen["result"]
    features_used = chosen["features"]
    
    # Prepare plot data
    vis_sample = df[selected_features]
    if vis_sample.shape[0] > MAX_3D_POINTS:
        st.info(f"Pour la visualisation, l'affichage est échantillonné à {MAX_3D_POINTS} points.")
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
            f"{res['algo']} — Visualisation 3D ({sel})",
            centroids, features_used
        )
    else:
        fig = create_2d_scatter(
            plot_df, x_axis, y_axis,
            labels_plot,
            f"{res['algo']} — Visualisation 2D ({sel})",
            centroids, features_used
        )
    
    st.plotly_chart(fig, width='stretch')
    
    # Legend and help
    st.markdown(
        "**Légende** : chaque symbole/valeur de couleur représente un cluster. "
        "Le gris indique souvent le bruit (DBSCAN)."
    )
    
    with st.expander("Définitions rapides des métriques (clic pour ouvrir)"):
        st.markdown("""
**Silhouette Score** — mesure de cohésion / séparation. Valeur dans [-1,1], plus élevé = meilleure séparation.

**Calinski-Harabasz** — ratio variance inter / intra-cluster (plus élevé = mieux).

**Davies-Bouldin** — moyenne similarité intra- / inter-cluster (plus faible = mieux).

**Inertie (WCSS)** — somme des carrés des distances intra-cluster (plus faible = clusters plus compacts).
""")


def render_metrics_tab():
    """Render the metrics comparison tab."""
    st.subheader("Comparaison quantitative des runs stockés")
    
    runs = get_results_list()
    
    if len(runs) == 0:
        st.info("Aucun résultat stocké. Exécutez au moins un algorithme pour comparer.")
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
    st.markdown("### Tableau des métriques d'évaluation")
    st.markdown("*↑ = plus élevé = mieux, ↓ = plus faible = mieux*")
    st.dataframe(metrics_df, width='stretch', hide_index=True)
    
    # Display parameters table
    st.markdown("### Tableau des paramètres utilisés")
    st.dataframe(params_df, width='stretch', hide_index=True)
    
    # Side-by-side comparison
    st.markdown("---")
    st.markdown("### Comparaison side-by-side")
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
        st.markdown("**Métriques**")
        comp_metrics_df = pd.DataFrame({
            "Métrique": ["N° Clusters effectifs", "Silhouette ↑", "Calinski-Harabasz ↑", "Davies-Bouldin ↓", "Inertie (WCSS) ↓"],
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
        st.markdown("**Paramètres**")
        # Collect all unique parameter names
        all_params = set(pA.keys()) | set(pB.keys())
        param_rows = []
        for param in sorted(all_params):
            param_rows.append({
                "Paramètre": param,
                f"Run A ({rA['algo']})": pA.get(param, "N/A"),
                f"Run B ({rB['algo']})": pB.get(param, "N/A"),
            })
        if param_rows:
            comp_params_df = pd.DataFrame(param_rows)
            st.table(comp_params_df)
        else:
            st.info("Aucun paramètre à comparer.")


def render_charts_tab(df: pd.DataFrame, selected_features: list[str]):
    """
    Render the additional charts tab with algorithm-specific visualizations.
    
    Args:
        df: Full DataFrame
        selected_features: List of selected feature names
    """
    st.subheader("Graphiques complémentaires")
    
    runs = get_results_list()
    
    if len(runs) == 0:
        st.info("Exécutez des algorithmes pour générer ces graphiques.")
        return
    
    # Filter runs compatible with current dataset
    compatible_runs = get_compatible_runs(df)
    
    if len(compatible_runs) == 0:
        st.warning(
            "Aucun run compatible avec le dataset actuel. "
            "Les runs précédents utilisaient des features différentes. "
            "Exécutez un nouveau clustering sur ce dataset."
        )
        return
    
    # Run selection (only compatible runs)
    sel_run = st.selectbox(
        "Choisir un run pour les graphiques",
        options=compatible_runs,
        format_func=lambda k: f"{k} — {st.session_state['results'][k]['result']['algo']}"
    )
    
    chosen = st.session_state["results"][sel_run]
    res = chosen["result"]
    features_used = chosen["features"]
    algo = res["algo"]
    
    # Validate features exist in current dataframe
    missing_features = [f for f in features_used if f not in df.columns]
    if missing_features:
        st.error(f"Features manquantes dans le dataset actuel: {missing_features}")
        return
    
    X = df[features_used]
    X_np = X.to_numpy()

    # Dispatch charts based on algorithm
    if algo == "KMeans":
        _render_kmeans_charts(X_np, res, sel_run)
    elif algo == "K-Medoids":
        _render_kmedoids_charts(X_np, res, sel_run)
    elif algo == "DBSCAN":
        _render_dbscan_charts(X_np, res, sel_run)
    elif algo == "AGNES":
        _render_agnes_charts(X_np, res, sel_run)
    elif algo == "DIANA":
        _render_diana_charts(X_np, res, sel_run)
    else:
        # Generic chart for unknown algorithms
        _render_generic_charts(X_np, res, sel_run)


    


def _render_diana_charts(X: np.ndarray, res: dict, run_name: str):
    """Render DIANA specific charts: Dendrogram + Silhouette + Histogram"""
    tabs = st.tabs(["Dendrogramme", "Silhouette vs K", "Distribution des clusters"])
    
    # Dendrogram
    with tabs[0]:
        st.markdown("**Dendrogramme (DIANA - Divisif)** : affichage hiérarchique du clustering.")
        n_clusters = int(res.get("params", {}).get("n_clusters", 2))
        k_cut = st.slider("Couper pour obtenir combien de clusters ?", 2, min(10, X.shape[0]), n_clusters, key=f"diana_dendrogram_{run_name}")
        try:
            Z = compute_linkage_matrix(X, method="complete")
            fig = create_dendrogram(Z, k_cut, figsize=(12, 5))
            st.pyplot(fig)
        except Exception as e:
            st.error(f"Impossible de calculer le dendrogramme: {e}")
    
    # Silhouette plot
    with tabs[1]:
        st.markdown("**Score de Silhouette** : mesure de qualité vs nombre de clusters.")
        if st.button("Calculer Silhouette", key=f"diana_silhouette_{run_name}"):
            from clustering.diana import compute_diana_analysis
            metric = res.get("params", {}).get("metric", "euclidean")
            analysis = compute_diana_analysis(X, metric=metric, k_range=range(2, min(11, X.shape[0])))
            fig = create_silhouette_plot(analysis["k_values"], analysis["silhouette_scores"])
            fig.update_layout(title="Silhouette Score vs. K (DIANA)")
            st.plotly_chart(fig, use_container_width=True)
    
    # Histogram
    with tabs[2]:
        st.markdown("**Distribution des clusters** : nombre de points par cluster.")
        labels = res.get("labels", np.array([]))
        fig = create_cluster_histogram(labels)
        st.plotly_chart(fig, use_container_width=True)


def _render_agnes_charts(X: np.ndarray, res: dict, run_name: str):
    """Render AGNES specific charts: Dendrogram + Silhouette + Histogram"""
    tabs = st.tabs(["Dendrogramme", "Silhouette vs K", "Distribution des clusters"])
    
    # Dendrogram
    with tabs[0]:
        st.markdown("**Dendrogramme (AGNES - Agglomératif)** : affichage hiérarchique du clustering.")
        n_clusters = int(res.get("params", {}).get("n_clusters", 2))
        linkage_method = res.get("params", {}).get("linkage", "ward")
        k_cut = st.slider("Couper pour obtenir combien de clusters ?", 2, min(10, X.shape[0]), n_clusters, key=f"agnes_dendrogram_{run_name}")
        try:
            Z = compute_linkage_matrix(X, method=linkage_method)
            fig = create_dendrogram(Z, k_cut, figsize=(12, 5))
            st.pyplot(fig)
        except Exception as e:
            st.error(f"Impossible de calculer le dendrogramme: {e}")
    
    # Silhouette plot
    with tabs[1]:
        st.markdown("**Score de Silhouette** : mesure de qualité vs nombre de clusters.")
        if st.button("Calculer Silhouette", key=f"agnes_silhouette_{run_name}"):
            from clustering.agnes import compute_agnes_analysis
            linkage_method = res.get("params", {}).get("linkage", "ward")
            analysis = compute_agnes_analysis(X, method=linkage_method, k_range=range(2, min(11, X.shape[0])))
            fig = create_silhouette_plot(analysis["k_values"], analysis["silhouette_scores"])
            fig.update_layout(title="Silhouette Score vs. K (AGNES)")
            st.plotly_chart(fig, use_container_width=True)
    
    # Histogram
    with tabs[2]:
        st.markdown("**Distribution des clusters** : nombre de points par cluster.")
        labels = res.get("labels", np.array([]))
        fig = create_cluster_histogram(labels)
        st.plotly_chart(fig, use_container_width=True)


def _render_kmeans_charts(X: np.ndarray, res: dict, run_name: str):
    """Render K-Means specific charts: Elbow + Silhouette + Histogram"""
    tabs = st.tabs(["Elbow Plot", "Silhouette vs K", "Distribution des clusters"])

    # Elbow plot
    with tabs[0]:
        st.markdown("**Courbe d'Elbow** : inertie en fonction du nombre de clusters.")
        if st.button("Calculer Elbow", key=f"kmeans_elbow_{run_name}"):
            k_values, inertias = compute_elbow(X, range(2, min(11, X.shape[0])))
            fig = create_elbow_plot(k_values, inertias)
            st.plotly_chart(fig, use_container_width=True)

    # Silhouette plot
    with tabs[1]:
        st.markdown("**Score de Silhouette** : mesure de qualité vs nombre de clusters.")
        if st.button("Calculer Silhouette", key=f"kmeans_silhouette_{run_name}"):
            k_values, scores = compute_silhouette_scores(X, range(2, min(11, X.shape[0])))
            fig = create_silhouette_plot(k_values, scores)
            st.plotly_chart(fig, use_container_width=True)

    # Histogram
    with tabs[2]:
        st.markdown("**Distribution des clusters** : nombre de points par cluster.")
        labels = res.get("labels", np.array([]))
        fig = create_cluster_histogram(labels)
        st.plotly_chart(fig, use_container_width=True)


def _render_kmedoids_charts(X: np.ndarray, res: dict, run_name: str):
    """Render K-Medoids specific charts: Elbow + Silhouette + Histogram"""
    tabs = st.tabs(["Elbow Plot", "Silhouette vs K", "Distribution des clusters"])

    # Elbow plot for K-Medoids
    with tabs[0]:
        st.markdown("**Courbe d'Elbow (K-Medoids)** : inertie en fonction du nombre de clusters.")
        metric = res.get("params", {}).get("metric", "euclidean")
        if st.button("Calculer Elbow", key=f"kmedoids_elbow_{run_name}"):
            from clustering.kmedoids import compute_kmedoids_elbow
            k_values, inertias = compute_kmedoids_elbow(X, range(2, min(11, X.shape[0])), metric=metric)
            fig = create_elbow_plot(k_values, inertias)
            fig.update_layout(title="Courbe d'Elbow (K-Medoids)")
            st.plotly_chart(fig, use_container_width=True)

    # Silhouette plot
    with tabs[1]:
        st.markdown("**Score de Silhouette** : mesure de qualité vs nombre de clusters.")
        if st.button("Calculer Silhouette", key=f"kmedoids_silhouette_{run_name}"):
            k_values, scores = compute_silhouette_scores(X, range(2, min(11, X.shape[0])))
            fig = create_silhouette_plot(k_values, scores)
            st.plotly_chart(fig, use_container_width=True)

    # Histogram
    with tabs[2]:
        st.markdown("**Distribution des clusters** : nombre de points par cluster.")
        labels = res.get("labels", np.array([]))
        fig = create_cluster_histogram(labels)
        st.plotly_chart(fig, use_container_width=True)


def _render_dbscan_charts(X: np.ndarray, res: dict, run_name: str):
    """Render DBSCAN specific charts: k-Distance + Histogram"""
    tabs = st.tabs(["k-Distance Graph", "Distribution des clusters"])

    # k-Distance graph
    with tabs[0]:
        st.markdown("**k-Distance Graph** : pour déterminer le paramètre epsilon.")
        min_samples = int(res.get("params", {}).get("min_samples", 5))
        if st.button("Calculer k-Distance", key=f"dbscan_kdist_{run_name}"):
            distances = compute_kdistances(X, k=min_samples)
            fig = create_kdistance_graph(distances, k=min_samples)
            st.plotly_chart(fig, use_container_width=True)

    # Histogram
    with tabs[1]:
        st.markdown("**Distribution des clusters** : nombre de points par cluster (incluant bruit).")
        labels = res.get("labels", np.array([]))
        fig = create_cluster_histogram(labels)
        st.plotly_chart(fig, use_container_width=True)


def _render_generic_charts(X: np.ndarray, res: dict, run_name: str):
    """Render generic charts for any algorithm: Histogram"""
    st.markdown("**Distribution des clusters** : nombre de points par cluster.")
    labels = res.get("labels", np.array([]))
    fig = create_cluster_histogram(labels)
    st.plotly_chart(fig, use_container_width=True)

