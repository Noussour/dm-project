"""
Sidebar components for data loading and algorithm configuration.
"""

import numpy as np
import pandas as pd
import streamlit as st

from utils.data_loader import (
    read_uploaded_file,
    compute_five_number_summary,
    filter_dataframe,
)
from utils.preprocessing import PreprocessingPipeline
from components.preprocessing_ui import (
    render_outlier_detection_section,
    render_missing_values_strategy,
    render_normalization_section,
    run_preprocessing_pipeline,
)
from config.constants import SUPPORTED_ALGORITHMS


def render_data_preview(df: pd.DataFrame):
    """
    Render data preview expander with metadata and five-number summary.
    
    Args:
        df: DataFrame to preview
    """
    with st.sidebar.expander("Aperçu (5 premières lignes)"):
        st.write(df.head(5))
        st.markdown("---")
        
        # Instance and attribute counts
        st.markdown(f"**Nombre d'instances (lignes)** : {df.shape[0]}")
        st.markdown(f"**Nombre d'attributs (colonnes)** : {df.shape[1]}")
        
        # Attribute list with types
        attr_df = pd.DataFrame({
            "Nom": df.columns,
            "Type": [str(df[col].dtype) for col in df.columns],
        })
        st.markdown("**Attributs (nom & type)** :")
        st.dataframe(attr_df, width='stretch')
        
        # Five-number summary
        five_num = compute_five_number_summary(df)
        if five_num is not None:
            st.markdown("**Five-number summary (pour attributs numériques)** :")
            st.dataframe(five_num, width='stretch')
        else:
            st.info("Aucune colonne numérique détectée pour calculer les quantiles/five-number summary.")


def render_file_upload() -> tuple[pd.DataFrame | None, str | None]:
    """
    Render file upload widget and return loaded DataFrame.
    Users must upload their own CSV or Excel files.
    When a new dataset is uploaded, reset all preprocessing and clustering results.
    
    Returns:
        Tuple of (DataFrame, filename) or (None, None)
    """
    st.sidebar.header("📂 Chargement des données")
    st.sidebar.markdown("_Les données doivent être fournies par l'utilisateur via upload de fichier_")
    
    uploaded = st.sidebar.file_uploader(
        "📥 Uploader un fichier CSV ou Excel (xlsx/xls)",
        accept_multiple_files=False
    )
    
    df = None
    dataset_name = None
    
    if uploaded is not None:
        # Check if this is a new dataset (different from what was loaded before)
        current_dataset_key = st.session_state.get("current_uploaded_file")
        if current_dataset_key != uploaded.name:
            # New dataset uploaded - reset ALL session data
            st.session_state["current_uploaded_file"] = uploaded.name
            st.session_state["preprocessed_datasets"] = {}
            st.session_state["results"] = {}
            st.session_state["classification_results"] = {}
            st.session_state["selected_features_for_clustering"] = []
            st.session_state["selected_dataset_key"] = None
            st.session_state["active_section"] = "Prétraitement"
            st.toast("Nouveau dataset détecté — toutes les données de session ont été réinitialisées.")
        
        df = read_uploaded_file(uploaded)
        if df is not None:
            dataset_name = uploaded.name
            st.sidebar.success(
                f"Fichier '{uploaded.name}' chargé ({df.shape[0]} lignes, {df.shape[1]} colonnes)"
            )
        else:
            st.sidebar.warning("Impossible de lire le fichier uploadé.")
    
    # Show preview if data loaded
    if df is not None:
        render_data_preview(df)
    
    return df, dataset_name


def render_feature_selection(df: pd.DataFrame) -> list[str]:
    """
    Render feature selection multiselect.
    By default, selects all numeric features.
    
    Args:
        df: DataFrame with features
        
    Returns:
        List of selected feature names
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    st.sidebar.markdown("### Sélection des features")
    selected_features = st.sidebar.multiselect(
        "Choisissez au moins 1 feature numérique pour le clustering",
        numeric_cols,
        default=numeric_cols,  # Select all features by default
    )
    
    return selected_features

def render_preprocessing(df: pd.DataFrame, selected_features: list) -> tuple[pd.DataFrame, list[str]]:
    """
    Render data preprocessing options with outlier detection and advanced strategy selection.
    
    Args:
        df: Input DataFrame
        selected_features: Selected numeric features
        
    Returns:
        Tuple of (processed_DataFrame, list_of_applied_steps)
    """
    st.markdown("---")
    st.markdown("### Prétraitement des données")
    
    with st.expander("Configuration du pipeline", expanded=True):
        # Outlier detection and removal
        st.markdown("#### ☑️ Étape 1: Suppression des outliers")
        remove_outliers, outlier_features = render_outlier_detection_section(df, selected_features)
        
        st.markdown("---")
        
        # Missing values strategy
        st.markdown("#### ☑️ Étape 2: Gestion des valeurs manquantes")
        
        # Show missing values warning
        if df.isnull().values.any():
            missing_total = df.isnull().sum().sum()
            st.warning(f"Dataset contient **{missing_total}** valeur(s) manquante(s)")
            
            with st.expander("Voir les détails"):
                missing_counts = df.isnull().sum()
                for col, count in missing_counts[missing_counts > 0].items():
                    pct = (count / len(df)) * 100
                    st.write(f"**{col}**: {count} valeurs ({pct:.1f}%)")
        
        missing_strategy = render_missing_values_strategy(df)
        
        st.markdown("---")
        
        # Normalization
        st.markdown("#### ☑️ Étape 3: Normalisation des features")
        normalization_config = render_normalization_section(df)
        
        st.markdown("---")
        
        # Run preprocessing button
        # col1, col2 = st.columns([2, 1])
        # with col1:
        run_preprocessing_btn = st.button(
                "▶️ Exécuter le pipeline de prétraitement",
                key="run_preprocessing_btn",
                width='stretch'
            )
        
        if run_preprocessing_btn:
            df_processed = run_preprocessing_pipeline(
                df,
                selected_features,
                remove_outliers,
                outlier_features,
                missing_strategy,
                normalization_config
            )
            
            # Store in session state
            st.session_state["preprocessed_datasets"] = st.session_state.get("preprocessed_datasets", {})
            
            # Create meaningful dataset name based on preprocessing steps
            preprocessing_name = []
            
            if remove_outliers:
                preprocessing_name.append("NoOutliers")
            
            if missing_strategy.get("global_strategy") and missing_strategy["global_strategy"] != "ignore":
                global_strat = missing_strategy["global_strategy"]
                preprocessing_name.append(f"{global_strat.capitalize()}")
            
            if normalization_config.get("method"):
                norm_method = normalization_config["method"]
                preprocessing_name.append(norm_method.replace("-", "").capitalize())
            
            # Create dataset key with meaningful name
            timestamp = pd.Timestamp.now().strftime("%H%M%S")
            if preprocessing_name:
                dataset_key = f"[{' + '.join(preprocessing_name)}] {timestamp}"
            else:
                dataset_key = f"Original {timestamp}"
            
            st.session_state["preprocessed_datasets"][dataset_key] = df_processed
            st.session_state["last_preprocessed_key"] = dataset_key
            
            st.success(f"Dataset prétraité sauvegardé: `{dataset_key}`")
            
            return df_processed, ["Preprocessing executed"]
    
    # Return original df if no preprocessing was run
    return df, []

def render_algorithm_params() -> tuple[str, dict, bool, bool, str]:
    """
    Render algorithm selection, dataset selection, and parameter widgets.
    
    Returns:
        Tuple of (algorithm_name, parameters_dict, best_params_button, run_button, selected_dataset_key)
    """
    st.sidebar.markdown("---")
    st.sidebar.header("Paramètres de clustering")
    
    # Dataset selection
    selected_dataset_key = None
    if "preprocessed_datasets" in st.session_state and st.session_state["preprocessed_datasets"]:
        preprocessed_keys = list(st.session_state["preprocessed_datasets"].keys())
        selected_dataset_key = st.sidebar.selectbox(
            "Sélectionnez le dataset",
            preprocessed_keys,
            index=len(preprocessed_keys) - 1,
            format_func=lambda k: f"{k} (shape: {st.session_state['preprocessed_datasets'][k].shape})"
        )
    else:
        st.sidebar.info("Dataset uploadé (aucun prétraitement disponible)")
        selected_dataset_key = "uploaded"
    
    st.sidebar.markdown("---")
    
    algo_choice = st.sidebar.selectbox(
        "Sélectionnez l'algorithme",
        SUPPORTED_ALGORITHMS
    )
    
    algo_params = {}
    
    if algo_choice == "KMeans":
        st.sidebar.markdown("**K-Means** : partitionnement en k clusters (centroïdes)")
        algo_params["n_clusters"] = st.sidebar.slider("n_clusters (k)", 2, 10, 3)
        algo_params["init"] = st.sidebar.selectbox("Méthode d'initialisation", ("k-means++", "random"))
    
    elif algo_choice == "K-Medoids":
        st.sidebar.markdown("**K-Medoids (PAM)** : partitionnement avec médoïdes (robuste aux outliers)")
        algo_params["n_clusters"] = st.sidebar.slider("n_clusters (k)", 2, 10, 3)
        algo_params["metric"] = st.sidebar.selectbox("Métrique de distance", ("euclidean", "manhattan", "cosine"))
        algo_params["init"] = st.sidebar.selectbox("Méthode d'initialisation", ("k-medoids++", "heuristic", "random"))
        
    elif algo_choice == "DBSCAN":
        st.sidebar.markdown("**DBSCAN** : clustering basé sur la densité")
        algo_params["eps"] = st.sidebar.slider("eps (rayon)", 0.1, 5.0, 0.5, step=0.1)
        algo_params["min_samples"] = st.sidebar.slider("min_samples", 1, 20, 5)
        
    elif algo_choice == "AGNES":
        st.sidebar.markdown("**AGNES** : Agglomerative Nesting (approche ascendante)")
        algo_params["n_clusters"] = st.sidebar.slider("n_clusters", 2, 10, 2)
        algo_params["linkage"] = st.sidebar.selectbox("Méthode de liaison", ("ward", "complete", "average", "single"))
        algo_params["metric"] = st.sidebar.selectbox("Métrique de distance", ("euclidean", "manhattan", "cosine"))
    
    elif algo_choice == "DIANA":
        st.sidebar.markdown("**DIANA** : Divisive Analysis (approche descendante)")
        algo_params["n_clusters"] = st.sidebar.slider("n_clusters", 2, 10, 2)
        algo_params["metric"] = st.sidebar.selectbox("Métrique de distance", ("euclidean", "manhattan"))
    
    # Render action buttons below algorithm parameters
    st.sidebar.markdown("---")
    col1, col2 = st.sidebar.columns(2)
    with col1:
        best_params_button = st.button("🔍 Meilleurs params", use_container_width=True, key="btn_best_params_sidebar")
    with col2:
        run_button = st.button("▶️ Exécuter", use_container_width=True, key="btn_run_sidebar")
    
    st.sidebar.markdown("---")
    
    return algo_choice, algo_params, best_params_button, run_button, selected_dataset_key


def render_run_button() -> bool:
    """
    Render the clustering execution button.
    
    Returns:
        True if button was clicked
    """
    return st.sidebar.button("Exécuter le clustering", use_container_width=True)


def render_best_parameters_button(algo_choice: str, df: pd.DataFrame, selected_features: list) -> bool:
    """
    Render button to calculate and display best parameters.
    
    Returns:
        True if button was clicked
    """
    return st.sidebar.button("Calculer les meilleurs paramètres", use_container_width=True)


def calculate_best_parameters(algo_choice: str, df: pd.DataFrame, selected_features: list) -> dict:
    """
    Calculate and return best parameters for the given algorithm without displaying plots.
    
    Args:
        algo_choice: Selected clustering algorithm
        df: Data DataFrame
        selected_features: List of selected feature names
    
    Returns:
        Dictionary of best parameters for the algorithm
    """
    from sklearn.cluster import KMeans
    from sklearn.mixture import GaussianMixture
    from sklearn.metrics import silhouette_score
    from sklearn.neighbors import NearestNeighbors
    
    X = df[selected_features].values
    n_samples = X.shape[0]
    best_params = {}
    
    if algo_choice == "KMeans":
        # Find best k using silhouette score
        # Limit k to max number of samples
        max_k = min(10, max(2, n_samples - 1))
        silhouette_scores = []
        K_range = range(2, max_k + 1)
        
        for k in K_range:
            if k <= n_samples:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                kmeans.fit(X)
                silhouette_scores.append(silhouette_score(X, kmeans.labels_))
        
        if silhouette_scores:
            best_k = list(K_range)[np.argmax(silhouette_scores)]
            best_params["n_clusters"] = min(best_k, n_samples)
        else:
            best_params["n_clusters"] = min(2, n_samples)
        best_params["init"] = "k-means++"
    
    elif algo_choice == "DBSCAN":
        # Find best eps using k-distance graph
        # Adjust k to be less than number of samples
        k_neighbors = min(4, max(1, n_samples - 1))
        neighbors = NearestNeighbors(n_neighbors=k_neighbors)
        neighbors.fit(X)
        distances, _ = neighbors.kneighbors(X)
        distances = np.sort(distances[:, -1], axis=0)
        
        # Use 90th percentile as eps, or median if dataset is very small
        if len(distances) > 1:
            best_eps = float(distances[int(len(distances) * 0.9)])
        else:
            best_eps = float(distances[0]) if len(distances) > 0 else 0.5
        
        best_params["eps"] = best_eps
        best_params["min_samples"] = min(5, max(2, n_samples // 2))
    
    elif algo_choice == "K-Medoids":
        # Find best k using silhouette score with custom K-Medoids
        from clustering.kmedoids import kmedoids_pam
        silhouette_scores = []
        K_range = range(2, 11)
        
        for k in K_range:
            try:
                labels, _, _ = kmedoids_pam(X, k, metric="euclidean", random_state=42)
                silhouette_scores.append(silhouette_score(X, labels))
            except Exception:
                silhouette_scores.append(0)
        
        best_k = list(K_range)[np.argmax(silhouette_scores)]
        best_params["n_clusters"] = best_k
        best_params["metric"] = "euclidean"
    
    elif algo_choice == "AGNES":
        # Find best k using silhouette score
        from sklearn.cluster import AgglomerativeClustering
        silhouette_scores = []
        K_range = range(2, 11)
        
        for k in K_range:
            agnes = AgglomerativeClustering(n_clusters=k, linkage="ward")
            labels = agnes.fit_predict(X)
            silhouette_scores.append(silhouette_score(X, labels))
        
        best_k = list(K_range)[np.argmax(silhouette_scores)]
        best_params["n_clusters"] = best_k
        best_params["linkage"] = "ward"
        best_params["metric"] = "euclidean"
    
    elif algo_choice == "DIANA":
        # Find best k using silhouette score
        from clustering.diana import diana_clustering
        silhouette_scores = []
        K_range = range(2, 11)
        
        for k in K_range:
            try:
                labels, _ = diana_clustering(X, k, "euclidean")
                if len(np.unique(labels)) >= 2:
                    silhouette_scores.append(silhouette_score(X, labels))
                else:
                    silhouette_scores.append(0)
            except Exception:
                silhouette_scores.append(0)
        
        best_k = list(K_range)[np.argmax(silhouette_scores)]
        best_params["n_clusters"] = best_k
        best_params["metric"] = "euclidean"

    
    return best_params


def display_best_parameters_analysis(algo_choice: str, df: pd.DataFrame, selected_features: list):
    """
    Display plots and recommendations for best parameters based on algorithm.
    
    Args:
        algo_choice: Selected clustering algorithm
        df: Data DataFrame
        selected_features: List of selected feature names
    """
    import matplotlib.pyplot as plt
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    
    X = df[selected_features].values
    
    st.markdown("### Analyse des meilleurs paramètres")
    
    if algo_choice == "KMeans":
        # Elbow method
        st.markdown("#### Méthode du coude (Elbow Method)")
        inertias = []
        silhouette_scores = []
        K_range = range(2, 11)
        
        for k in K_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(X)
            inertias.append(kmeans.inertia_)
            silhouette_scores.append(silhouette_score(X, kmeans.labels_))
        
        # Create two-column layout for plots
        col1, col2 = st.columns(2)
        
        with col1:
            # Plot 1: Inertia (Elbow)
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.plot(K_range, inertias, 'bo-', linewidth=2, markersize=8)
            ax.set_xlabel("Nombre de clusters (k)", fontsize=12)
            ax.set_ylabel("Inertie", fontsize=12)
            ax.set_title("Méthode du coude", fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            st.info("Cherchez le 'coude' - le point où la courbe commence à s'aplatir")
        
        with col2:
            # Plot 2: Silhouette Score
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.plot(K_range, silhouette_scores, 'go-', linewidth=2, markersize=8)
            ax.set_xlabel("Nombre de clusters (k)", fontsize=12)
            ax.set_ylabel("Score de Silhouette", fontsize=12)
            ax.set_title("Score de Silhouette", fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            st.info("Plus le score est proche de 1, meilleur est le clustering")
        
        # Recommendation
        best_k = K_range[np.argmax(silhouette_scores)]
        st.success(f"**Valeur recommandée de k : {best_k}** (basée sur le score de silhouette)")
    
    elif algo_choice == "DBSCAN":
        st.markdown("#### Analyse de voisinage (k-distance graph)")
        from sklearn.neighbors import NearestNeighbors
        
        # Calculate k-distance graph
        neighbors = NearestNeighbors(n_neighbors=5)
        neighbors_fit = neighbors.fit(X)
        distances, indices = neighbors_fit.kneighbors(X)
        distances = np.sort(distances[:, -1], axis=0)
        
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(distances, linewidth=2)
        ax.set_ylabel("k-distance (k=5)", fontsize=12)
        ax.set_xlabel("Points triés", fontsize=12)
        ax.set_title("Graphique k-distance pour déterminer eps", fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        
        st.info("Cherchez le point d'inflexion (genou) - c'est une bonne valeur pour eps")
        st.success(f"**Valeur recommandée pour eps : ~{distances[int(len(distances)*0.9)]:.2f}** (90e percentile)")



def render_sidebar_footer():
    """Render sidebar footer with help and management options."""
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Gestion des runs")
    
    if st.sidebar.button("Réinitialiser les résultats stockés"):
        st.session_state["results"] = {}
        st.sidebar.success("Historique des runs réinitialisé.")
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Aide & références")
    st.sidebar.markdown("""
- **Silhouette** : [-1,1], plus élevé = meilleure séparation.
- **Calinski-Harabasz** : plus élevé = meilleur.
- **Davies-Bouldin** : plus faible = meilleur.
""")


def render_sidebar(df: pd.DataFrame = None) -> dict:
    """
    Render complete sidebar and return configuration.
    
    Args:
        df: Optional pre-loaded DataFrame
        
    Returns:
        Dictionary with sidebar configuration and data
    """
    config = {
        "df": None,
        "dataset_name": None,
        "selected_features": [],
        "algo_choice": None,
        "algo_params": {},
        "run_clicked": False,
    }
    
    # File upload
    if df is None:
        config["df"], config["dataset_name"] = render_file_upload()
    else:
        config["df"] = df
    
    if config["df"] is not None:
        config["selected_features"] = render_feature_selection(config["df"])
        
        if len(config["selected_features"]) >= 2:
            config["algo_choice"], config["algo_params"] = render_algorithm_params()
            config["run_clicked"] = render_run_button()
    
    render_sidebar_footer()
    
    return config
