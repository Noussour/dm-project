"""
Streamlit application to compare clustering and classification algorithms.

Execution: `streamlit run app.py`

Project Structure:
├── app.py              # Main entry point
├── config/             # Configuration and constants
│   ├── settings.py     # Streamlit configuration
│   └── constants.py    # Application constants
├── utils/              # Utility functions
│   ├── data_loader.py  # Data loading and validation
│   └── metrics.py      # Clustering metrics
├── clustering/         # Clustering algorithms
│   ├── algorithms.py   # Main orchestrator
│   ├── kmeans.py       # K-Means
│   ├── kmedoids.py     # K-Medoids (PAM)
│   ├── dbscan.py       # DBSCAN
│   ├── agnes.py        # AGNES (Agglomerative Nesting)
│   └── diana.py        # DIANA (Divisive Analysis)
├── classification/     # Classification algorithms
│   ├── algorithms.py   # Main orchestrator
│   ├── knn.py          # k-Nearest Neighbors
│   ├── naive_bayes.py  # Naive Bayes
│   ├── decision_tree.py # C4.5 Decision Tree
│   ├── svm.py          # Support Vector Machine
│   └── metrics.py      # Classification metrics
├── visualization/      # Visualization
│   ├── plots.py        # Plotting functions
│   └── colors.py       # Color management
└── components/         # Streamlit UI Components
    ├── sidebar.py      # Clustering sidebar
    ├── classification_sidebar.py  # Classification sidebar
    ├── tabs.py         # Clustering tabs
    └── classification_tabs.py     # Classification tabs
"""

from datetime import datetime

import numpy as np
import streamlit as st

# Local imports
from config.settings import configure_page
from config.constants import MAX_3D_POINTS, ALGORITHM_CONSTRAINTS
from utils.data_loader import (
    validate_dataframe, 
    get_numeric_columns, 
    filter_dataframe,
    validate_algorithm_compatibility,
    validate_clustering_params,
)
from clustering import run_clustering
from components.sidebar import (
    render_file_upload,
    render_feature_selection,
    render_preprocessing,
    render_algorithm_params,
    render_run_button,
    render_best_parameters_button,
    display_best_parameters_analysis,
    calculate_best_parameters,
    render_sidebar_footer,
)
from components.tabs import (
    render_visualization_tab,
    render_metrics_tab,
    render_charts_tab,
)
from components.classification_sidebar import (
    render_target_selection,
    render_feature_selection_classification,
    render_train_test_split,
    render_classifier_params,
    render_classification_footer,
)
from components.classification_tabs import (
    render_classification_results_tab,
    render_all_classifiers_comparison,
)


def init_session_state():
    """Initialize Streamlit session state."""
    if "results" not in st.session_state:
        st.session_state["results"] = {}
    if "current_uploaded_file" not in st.session_state:
        st.session_state["current_uploaded_file"] = None
    if "preprocessed_datasets" not in st.session_state:
        st.session_state["preprocessed_datasets"] = {}
    if "selected_dataset_key" not in st.session_state:
        st.session_state["selected_dataset_key"] = None
    if "active_section" not in st.session_state:
        st.session_state["active_section"] = "Preprocessing"
    if "classification_results" not in st.session_state:
        st.session_state["classification_results"] = {}


def execute_clustering(df, selected_features, algo_choice, algo_params, dataset_name, use_best_params=False):
    """
    Execute clustering algorithm and store results.
    
    Args:
        df: DataFrame with data
        selected_features: List of feature column names
        algo_choice: Algorithm name
        algo_params: Algorithm parameters dictionary
        dataset_name: Name of the dataset file
        use_best_params: Whether best parameters were automatically detected
    """
    X = df[selected_features].copy()
    
    # Validate all features are numeric
    if not all(np.issubdtype(X[c].dtype, np.number) for c in X.columns):
        st.error("❌ All selected features must be numeric.")
        return
    
    # Validate algorithm compatibility with dataset
    is_valid, validation_messages = validate_algorithm_compatibility(
        df, selected_features, algo_choice
    )
    
    if not is_valid:
        for msg in validation_messages:
            if "valeur(s) manquante(s)" in msg.lower() or "missing value" in msg.lower():
                st.error(f"❌ {msg}")
            else:
                st.error(f"❌ {msg}")
        return
    
    # Show warnings if any
    for msg in validation_messages:
        if "valeur(s) manquante(s)" not in msg.lower() and "missing value" not in msg.lower():
            st.warning(f"⚠️ {msg}")
    
    # Validate clustering parameters against data size
    is_param_valid, param_error = validate_clustering_params(
        algo_choice, algo_params, len(df)
    )
    
    if not is_param_valid:
        st.error(f"❌ {param_error}")
        return
    
    try:
        st.info(f"🚀 Launching {algo_choice} ...")
        result = run_clustering(algo_choice, algo_params, X)
        
        # Create clean run identifier
        run_counter = len(st.session_state["results"]) + 1
        
        # Build parameter string based on algorithm
        param_str = ""
        if algo_choice == "KMeans":
            k = algo_params.get("n_clusters", 3)
            param_str = f"k{k}"
        elif algo_choice == "K-Medoids":
            k = algo_params.get("n_clusters", 3)
            param_str = f"k{k}"
        elif algo_choice == "DBSCAN":
            eps = algo_params.get("eps", 0.5)
            min_samples = algo_params.get("min_samples", 5)
            param_str = f"eps{eps:.1f}_min{min_samples}"
        elif algo_choice == "AGNES":
            n_clusters = algo_params.get("n_clusters", 2)
            linkage = algo_params.get("linkage", "ward")
            param_str = f"n{n_clusters}_{linkage}"
        elif algo_choice == "DIANA":
            n_clusters = algo_params.get("n_clusters", 2)
            param_str = f"n{n_clusters}"
        
        # Add auto indicator if best params were used
        auto_indicator = "Auto_" if use_best_params else ""
        
        # Create clean identifier: Run#_Algorithm_params
        ident = f"Run{run_counter}_{auto_indicator}{algo_choice}_{param_str}"
        
        # Store results
        st.session_state["results"][ident] = {
            "result": result,
            "features": selected_features,
            "dataset": dataset_name,
        }
        
        st.success(f"{algo_choice} executed - results: {ident}")
        
    except Exception as e:
        st.error(f"Error during execution: {e}")


def main():
    """Main application entry point."""
    configure_page()
    init_session_state()

    # Ensure active section default
    st.session_state.setdefault("active_section", "Preprocessing")

    # File upload + validation
    df, dataset_name = render_file_upload()
    if df is None:
        st.warning("Please load a dataset via the sidebar (file or predefined dataset).")
        st.stop()

    is_valid, error_msg = validate_dataframe(df)
    if not is_valid:
        st.error(error_msg)
        st.stop()

    # Default UI state values
    selected_features = []
    algo_choice = None
    algo_params = {}
    best_params_clicked = False
    run_clicked = False
    classifier_choice = None
    classifier_params = {}
    classify_clicked = False
    compare_clicked = False
    target_col = None
    split_config = {}

    # Sidebar behavior per active section
    # Sidebar behavior per active section
    if st.session_state["active_section"] == "Prétraitement" or st.session_state["active_section"] == "Preprocessing":
        # Keep internal state key as "Prétraitement" if that's what's used, or migrate it.
        # But for now let's just assume we translate the string value in the session state later or just handle both.
        # Ideally, we should unify the key, but "active_section" is used as a string.
        # Let's change the comparison to English if we change the state value.
        pass # I will handle the session state value change logic in the Navigation section below to avoid mismatch.
        # Actually, let's just stick to the current French keys for logic if we don't want to break things, 
        # BUT the user wants 100% translation. So I should probably change the keys too or at least the display.
        # Only the UI text needs to be English. The internal keys can remain French if it's too risky, 
        # but "active_section" value is used for logic AND display usually?
        # Let's check lines 265-275. Buttons set `st.session_state["active_section"]`.
        # So I will translate the values: "Preprocessing", "Clustering", "Classification".
        
    if st.session_state["active_section"] == "Preprocessing":
        selected_features = render_feature_selection(df)
        if len(selected_features) < 1:
            st.sidebar.error("Select at least 1 numeric feature.")
            st.stop()

    elif st.session_state["active_section"] == "Clustering":
        if "selected_features_for_clustering" not in st.session_state:
            st.session_state["selected_features_for_clustering"] = [c for c in df.columns if np.issubdtype(df[c].dtype, np.number)]

        selected_features = st.session_state["selected_features_for_clustering"]
        algo_choice, algo_params, best_params_clicked, run_clicked, selected_dataset_key = render_algorithm_params()
        st.session_state["selected_dataset_key"] = selected_dataset_key

    elif st.session_state["active_section"] == "Classification":
        df_for_classification = df.copy()
        if st.session_state.get("preprocessed_datasets"):
            keys = list(st.session_state["preprocessed_datasets"].keys())
            if keys:
                df_for_classification = st.session_state["preprocessed_datasets"][keys[-1]]

        target_col = render_target_selection(df_for_classification)
        if target_col is None:
            st.sidebar.error("Select a target variable.")
            st.stop()

        selected_features = render_feature_selection_classification(df_for_classification, target_col)
        if len(selected_features) == 0:
            st.sidebar.error("Select at least one feature.")
            st.stop()

        split_config = render_train_test_split()
        classifier_choice, classifier_params, classify_clicked, compare_clicked = render_classifier_params()
        render_classification_footer()

    # Navigation buttons
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Navigation")
    nav_col1, nav_col2, nav_col3 = st.sidebar.columns(3)

    with nav_col1:
        if st.button("Preprocess", use_container_width=True, key="nav_preprocess", disabled=st.session_state["active_section"] == "Preprocessing"):
            st.session_state["active_section"] = "Preprocessing"
            st.rerun()
    with nav_col2:
        if st.button("Cluster", use_container_width=True, key="nav_clustering", disabled=st.session_state["active_section"] == "Clustering"):
            st.session_state["active_section"] = "Clustering"
            st.rerun()
    with nav_col3:
        if st.button("Classify", use_container_width=True, key="nav_classification", disabled=st.session_state["active_section"] == "Classification"):
            st.session_state["active_section"] = "Classification"
            st.rerun()

    if st.sidebar.button("Reset", use_container_width=True, key="btn_reset_session"):
        st.session_state.clear()
        st.cache_data.clear()
        st.rerun()

    # Main content dispatch
    df_processed = df.copy()

    if st.session_state["active_section"] == "Preprocessing":
        st.title("Data Preprocessing")
        st.markdown("*Educational Data Mining Application*")
        st.markdown("---")

        current_dataset = df.copy()
        if st.session_state.get("preprocessed_datasets"):
            keys = list(st.session_state["preprocessed_datasets"].keys())
            if keys:
                current_dataset = st.session_state["preprocessed_datasets"][keys[-1]]

        if current_dataset.isnull().values.any():
            missing_count = current_dataset.isnull().sum().sum()
            st.warning(f"The dataset contains **{missing_count}** missing value(s). Use the options below to clean the dataset.")

        st.session_state["selected_features_for_clustering"] = selected_features
        df_processed, _ = render_preprocessing(df, selected_features)

        if st.session_state.get("preprocessed_datasets"):
            st.markdown("### Available Preprocessed Datasets")
            keys = list(st.session_state["preprocessed_datasets"].keys())
            selected_preprocessed = st.selectbox("Select a preprocessed dataset", keys, index=len(keys) - 1, key="select_preprocessed_dataset", format_func=lambda k: f"{k} (shape: {st.session_state['preprocessed_datasets'][k].shape})")
            if selected_preprocessed:
                df_processed = st.session_state["preprocessed_datasets"][selected_preprocessed]
                st.success(f"Using dataset: `{selected_preprocessed}`")

    elif st.session_state["active_section"] == "Clustering":
        st.title("Clustering Algorithms Comparator")
        st.markdown("*Data Mining: K-Means, K-Medoids, DBSCAN, AGNES, DIANA*")

        df_processed = df.copy()
        selected_dataset_name = "uploaded dataset"
        selected_dataset_key = st.session_state.get("selected_dataset_key")
        if st.session_state.get("preprocessed_datasets"):
            if selected_dataset_key and selected_dataset_key in st.session_state["preprocessed_datasets"]:
                df_processed = st.session_state["preprocessed_datasets"][selected_dataset_key]
                selected_dataset_name = selected_dataset_key
            else:
                keys = list(st.session_state["preprocessed_datasets"].keys())
                df_processed = st.session_state["preprocessed_datasets"][keys[-1]]
                selected_dataset_name = keys[-1]

        # Check for missing values and block execution if present
        has_missing_values = df_processed[selected_features].isnull().values.any() if selected_features else df_processed.isnull().values.any()
        if has_missing_values:
            missing_count = df_processed[selected_features].isnull().sum().sum() if selected_features else df_processed.isnull().sum().sum()
            st.error(f"❌ The dataset contains **{missing_count}** missing value(s) in the selected features. "
                    "Clustering algorithms cannot work with missing values.")
            st.info("💡 **Solution**: Return to the **Preprocessing** section and use a strategy to handle missing values "
                   "(deletion, imputation by mean/median/mode, etc.)")
            # Block execution but still show tabs for viewing previous results
            tabs = st.tabs(["2D/3D Visualization", "Metrics", "Charts"])
            with tabs[0]:
                render_visualization_tab(df_processed, selected_features)
            with tabs[1]:
                render_metrics_tab()
            with tabs[2]:
                render_charts_tab(df_processed, selected_features)
            st.stop()

        if df_processed.shape[0] > MAX_3D_POINTS:
            st.sidebar.info(f"For 3D visualization, sampling to {MAX_3D_POINTS} points.")

        if best_params_clicked:
            best_algo_params = calculate_best_parameters(algo_choice, df_processed, selected_features)
            st.info(f"**Best parameters for {algo_choice}:** {best_algo_params}")
            execute_clustering(df_processed, selected_features, algo_choice, best_algo_params, dataset_name, use_best_params=True)
        elif run_clicked:
            execute_clustering(df_processed, selected_features, algo_choice, algo_params, dataset_name, use_best_params=False)

        tabs = st.tabs(["2D/3D Visualization", "Metrics", "Charts"])
        with tabs[0]:
            render_visualization_tab(df_processed, selected_features)
        with tabs[1]:
            render_metrics_tab()
        with tabs[2]:
            render_charts_tab(df_processed, selected_features)

    elif st.session_state["active_section"] == "Classification":
        st.title("Supervised Classification")
        st.markdown("*Data Mining: k-NN, Naive Bayes, C4.5, SVM*")

        df_processed = df.copy()
        if st.session_state.get("preprocessed_datasets"):
            keys = list(st.session_state["preprocessed_datasets"].keys())
            selected_preprocessed = st.selectbox("Select a preprocessed dataset", keys, index=len(keys) - 1, key="select_preprocessed_dataset_classification", format_func=lambda k: f"{k} (shape: {st.session_state['preprocessed_datasets'][k].shape})")
            if selected_preprocessed:
                df_processed = st.session_state["preprocessed_datasets"][selected_preprocessed]

        available_numeric_cols = [c for c in df_processed.columns if np.issubdtype(df_processed[c].dtype, np.number)]
        selected_features = [f for f in selected_features if f in df_processed.columns]
        if not selected_features:
            selected_features = available_numeric_cols
            if target_col in selected_features:
                selected_features.remove(target_col)

        if target_col not in df_processed.columns:
            st.error(f"The target variable '{target_col}' does not exist in the preprocessed dataset. Please return to preprocessing or select another dataset.")
            st.stop()

        if df_processed.isnull().values.any():
            missing_count = df_processed.isnull().sum().sum()
            st.warning(f"The dataset contains **{missing_count}** missing value(s).")

        st.markdown("---")

        if compare_clicked:
            render_all_classifiers_comparison(df_processed, selected_features, target_col, split_config)
        elif classify_clicked:
            render_classification_results_tab(df_processed, selected_features, target_col, split_config, classifier_choice, classifier_params)
        else:
            st.info("""
            **Instructions:**
            1. Select the target variable (class) in the sidebar
            2. Configure the partition (80% training, 20% test)
            3. Choose a classification algorithm
            4. Click on **Classify** to execute, or **Compare All** to compare all algorithms
            
            **Available Algorithms:**
            - **k-NN** : k Nearest Neighbors (k = 1 to 10)
            - **Naive Bayes** : Naive Bayes Classifier
            - **C4.5** : Decision Tree (information gain)
            - **SVM** : Support Vector Machine
            """)


if __name__ == "__main__":
    main()
