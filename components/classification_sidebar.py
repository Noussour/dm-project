"""
Sidebar components for classification configuration.
"""

import numpy as np
import pandas as pd
import streamlit as st

from config.constants import SUPPORTED_CLASSIFIERS, DEFAULT_TEST_SIZE, KNN_K_RANGE


def render_target_selection(df: pd.DataFrame) -> str | None:
    """
    Render target variable selection.
    
    Args:
        df: DataFrame with all columns
        
    Returns:
        Selected target column name or None
    """
    all_cols = df.columns.tolist()
    
    if len(all_cols) == 0:
        st.sidebar.warning("The dataset is empty.")
        return None
    
    st.sidebar.markdown("### Target Selection")
    
    target_col = st.sidebar.selectbox(
        "Target Variable (Class)",
        all_cols,
        help="Select the column containing class labels (discrete values)"
    )
    
    # Validate target column
    if target_col:
        target_values = df[target_col].dropna()
        n_unique = target_values.nunique()
        
        # Check if target is continuous (too many unique values relative to dataset size)
        is_likely_continuous = False
        if np.issubdtype(df[target_col].dtype, np.floating):
            # Float column - check if values look continuous
            if n_unique > 20 or n_unique > len(df) * 0.5:
                is_likely_continuous = True
            # Also check if values have decimals
            elif target_values.apply(lambda x: x != int(x) if pd.notna(x) else False).any():
                is_likely_continuous = True
        
        if is_likely_continuous:
            st.sidebar.error(
                f"⚠️ The column '{target_col}' seems to contain continuous values "
                f"({n_unique} unique values). Classification requires discrete classes. "
                "Choose another column or discretize your data."
            )
            return None
        
        # Warning for too many classes
        if n_unique > 50:
            st.sidebar.warning(
                f"⚠️ The column '{target_col}' contains {n_unique} different classes. "
                "A large number of classes can affect performance."
            )
        
        # Show class distribution
        with st.sidebar.expander("Class Distribution"):
            class_counts = df[target_col].value_counts()
            st.bar_chart(class_counts)
            st.write(f"Number of classes: {len(class_counts)}")
            
            # Warn about imbalanced classes
            if len(class_counts) >= 2:
                max_count = class_counts.max()
                min_count = class_counts.min()
                if max_count > min_count * 10:
                    st.warning(
                        f"⚠️ Imbalanced classes: the largest class ({max_count}) "
                        f"is {max_count/min_count:.1f}x larger than the smallest ({min_count})."
                    )
    
    return target_col


def render_feature_selection_classification(df: pd.DataFrame, target_col: str) -> list:
    """
    Render feature selection for classification (excludes target column).
    
    Args:
        df: DataFrame with all columns
        target_col: The target column name (to exclude)
        
    Returns:
        List of selected feature column names
    """
    import numpy as np
    
    # Get numeric columns excluding target
    numeric_cols = [c for c in df.columns if c != target_col and np.issubdtype(df[c].dtype, np.number)]
    
    if len(numeric_cols) == 0:
        st.sidebar.warning("No numeric column available as feature.")
        return []
    
    st.sidebar.markdown("### Feature Selection")
    
    selected_features = st.sidebar.multiselect(
        "Features for classification",
        numeric_cols,
        default=numeric_cols,
        help="Select numeric columns to use as features"
    )
    
    return selected_features


def render_train_test_split() -> dict:
    """
    Render train/test split configuration.
    
    Returns:
        Dictionary with split configuration
    """
    st.sidebar.markdown("### Data Splitting")
    
    test_size = st.sidebar.slider(
        "Test Set Size (%)",
        min_value=10,
        max_value=50,
        value=int(DEFAULT_TEST_SIZE * 100),
        step=5,
        help="Recommended: 20% for testing and 80% for training"
    )
    
    random_state = st.sidebar.number_input(
        "Random Seed (Reproducibility)",
        min_value=0,
        max_value=1000,
        value=42,
        help="To ensure reproducible results"
    )
    
    stratify = st.sidebar.checkbox(
        "Stratification",
        value=True,
        help="Maintain class proportions in sets"
    )
    
    return {
        "test_size": test_size / 100,
        "random_state": int(random_state),
        "stratify": stratify,
    }


def render_classifier_params() -> tuple[str, dict]:
    """
    Render classifier selection and parameter widgets.
    
    Returns:
        Tuple of (classifier_name, parameters_dict)
    """
    st.sidebar.markdown("---")
    st.sidebar.header("🤖 Classifier Parameters")
    
    classifier_choice = st.sidebar.selectbox(
        "Select Algorithm",
        SUPPORTED_CLASSIFIERS
    )
    
    classifier_params = {}
    
    if classifier_choice == "k-NN":
        st.sidebar.markdown("**k-NN** : Classification by k-nearest neighbors")
        
        # Single k or range mode
        knn_mode = st.sidebar.radio(
            "Evaluation Mode",
            ["Single k", "Evaluate k from 1 to 10"],
            help="Evaluate k from 1 to 10 to find the best value"
        )
        
        if knn_mode == "Single k" or knn_mode == "Un seul k":
            classifier_params["k"] = st.sidebar.slider("k (number of neighbors)", 1, 20, 5)
            classifier_params["evaluate_range"] = False
        else:
            classifier_params["k_range"] = list(KNN_K_RANGE)
            classifier_params["evaluate_range"] = True
            st.sidebar.info("k will be evaluated from 1 to 10")
        
        classifier_params["metric"] = st.sidebar.selectbox(
            "Distance Metric",
            ("euclidean", "manhattan", "minkowski")
        )
        classifier_params["weights"] = st.sidebar.selectbox(
            "Weighting",
            ("uniform", "distance"),
            help="'uniform': all neighbors equal, 'distance': weighted by inverse of distance"
        )
        
    elif classifier_choice == "Naive Bayes":
        st.sidebar.markdown("**Naive Bayes** : Probabilistic Bayesian classifier")
        
        classifier_params["type"] = st.sidebar.selectbox(
            "Naive Bayes Type",
            ("gaussian", "multinomial", "bernoulli"),
            help="Gaussian: for continuous features, Multinomial: for counts, Bernoulli: for binary"
        )
        
        if classifier_params["type"] == "gaussian":
            classifier_params["var_smoothing"] = st.sidebar.slider(
                "var_smoothing",
                min_value=1e-12,
                max_value=1e-6,
                value=1e-9,
                format="%.1e",
                help="Portion of the largest variance added for stability"
            )
        else:
            classifier_params["alpha"] = st.sidebar.slider(
                "alpha (Laplace smoothing)",
                min_value=0.0,
                max_value=2.0,
                value=1.0,
                step=0.1
            )
        
    elif classifier_choice == "C4.5":
        st.sidebar.markdown("**C4.5** : Decision Tree (Information Gain)")
        
        classifier_params["criterion"] = st.sidebar.selectbox(
            "Split Criterion",
            ("entropy", "gini"),
            help="'entropy' corresponds to information gain (C4.5), 'gini' is used by CART"
        )
        
        max_depth_enabled = st.sidebar.checkbox("Limit Depth", value=False)
        if max_depth_enabled:
            classifier_params["max_depth"] = st.sidebar.slider("Max Depth", 1, 20, 5)
        else:
            classifier_params["max_depth"] = None
        
        classifier_params["min_samples_split"] = st.sidebar.slider(
            "min_samples_split",
            min_value=2,
            max_value=20,
            value=2,
            help="Minimum samples to split a node"
        )
        
        classifier_params["min_samples_leaf"] = st.sidebar.slider(
            "min_samples_leaf",
            min_value=1,
            max_value=10,
            value=1,
            help="Minimum samples in a leaf"
        )
        
    elif classifier_choice == "SVM":
        st.sidebar.markdown("**SVM** : Support Vector Machine")
        
        classifier_params["kernel"] = st.sidebar.selectbox(
            "Kernel Type",
            ("rbf", "linear", "poly", "sigmoid"),
            help="RBF is widely used for non-linear data"
        )
        
        classifier_params["C"] = st.sidebar.slider(
            "C (regularization)",
            min_value=0.01,
            max_value=100.0,
            value=1.0,
            help="Larger C means less tolerance for errors"
        )
        
        if classifier_params["kernel"] == "rbf":
            gamma_option = st.sidebar.selectbox(
                "gamma",
                ("scale", "auto", "manual"),
                help="RBF kernel coefficient"
            )
            if gamma_option == "manual" or gamma_option == "manuel":
                classifier_params["gamma"] = st.sidebar.slider("Gamma Value", 0.001, 10.0, 1.0)
            else:
                classifier_params["gamma"] = gamma_option
        
        if classifier_params["kernel"] == "poly":
            classifier_params["degree"] = st.sidebar.slider("Polynomial Degree", 2, 5, 3)
        
        classifier_params["normalize"] = st.sidebar.checkbox(
            "Normalize Features",
            value=True,
            help="Recommended for SVM"
        )
    
    # Action buttons
    st.sidebar.markdown("---")
    col1, col2 = st.sidebar.columns(2)
    with col1:
        run_button = st.button("▶️ Classify", use_container_width=True, key="btn_classify")
    with col2:
        compare_button = st.button("Compare All", use_container_width=True, key="btn_compare_all")
    
    return classifier_choice, classifier_params, run_button, compare_button


def render_classification_footer():
    """Render classification help in sidebar footer."""
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Classification Help")
    st.sidebar.markdown("""
**Metrics :**
- **Accuracy** : TP / (TP + FP)
- **Recall** : TP / (TP + FN)
- **F-measure** : 2 × (P × R) / (P + R)

**Confusion Matrix:**
- **TP** : True Positives
- **TN** : True Negatives
- **FP** : False Positives
- **FN** : False Negatives
""")
