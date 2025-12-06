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
        st.sidebar.warning("Le dataset est vide.")
        return None
    
    st.sidebar.markdown("### Sélection de la cible")
    
    target_col = st.sidebar.selectbox(
        "Variable cible (classe)",
        all_cols,
        help="Sélectionnez la colonne contenant les étiquettes de classe (valeurs discrètes)"
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
                f"⚠️ La colonne '{target_col}' semble contenir des valeurs continues "
                f"({n_unique} valeurs uniques). La classification nécessite des classes discrètes. "
                "Choisissez une autre colonne ou discrétisez vos données."
            )
            return None
        
        # Warning for too many classes
        if n_unique > 50:
            st.sidebar.warning(
                f"⚠️ La colonne '{target_col}' contient {n_unique} classes différentes. "
                "Un grand nombre de classes peut affecter les performances."
            )
        
        # Show class distribution
        with st.sidebar.expander("Distribution des classes"):
            class_counts = df[target_col].value_counts()
            st.bar_chart(class_counts)
            st.write(f"Nombre de classes: {len(class_counts)}")
            
            # Warn about imbalanced classes
            if len(class_counts) >= 2:
                max_count = class_counts.max()
                min_count = class_counts.min()
                if max_count > min_count * 10:
                    st.warning(
                        f"⚠️ Classes déséquilibrées: la plus grande classe ({max_count}) "
                        f"est {max_count/min_count:.1f}x plus grande que la plus petite ({min_count})."
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
        st.sidebar.warning("Aucune colonne numérique disponible comme feature.")
        return []
    
    st.sidebar.markdown("### Sélection des features")
    
    selected_features = st.sidebar.multiselect(
        "Features pour la classification",
        numeric_cols,
        default=numeric_cols,
        help="Sélectionnez les colonnes numériques à utiliser comme features"
    )
    
    return selected_features


def render_train_test_split() -> dict:
    """
    Render train/test split configuration.
    
    Returns:
        Dictionary with split configuration
    """
    st.sidebar.markdown("### Partitionnement des données")
    
    test_size = st.sidebar.slider(
        "Taille de l'ensemble de test (%)",
        min_value=10,
        max_value=50,
        value=int(DEFAULT_TEST_SIZE * 100),
        step=5,
        help="Recommandé: 20% pour le test et 80% pour l'apprentissage"
    )
    
    random_state = st.sidebar.number_input(
        "Seed aléatoire (reproductibilité)",
        min_value=0,
        max_value=1000,
        value=42,
        help="Pour garantir des résultats reproductibles"
    )
    
    stratify = st.sidebar.checkbox(
        "Stratification",
        value=True,
        help="Maintenir la proportion des classes dans les ensembles"
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
    st.sidebar.header("🤖 Paramètres du classifieur")
    
    classifier_choice = st.sidebar.selectbox(
        "Sélectionnez l'algorithme",
        SUPPORTED_CLASSIFIERS
    )
    
    classifier_params = {}
    
    if classifier_choice == "k-NN":
        st.sidebar.markdown("**k-NN** : Classification par k plus proches voisins")
        
        # Single k or range mode
        knn_mode = st.sidebar.radio(
            "Mode d'évaluation",
            ["Un seul k", "Évaluer k de 1 à 10"],
            help="Évaluer k de 1 à 10 pour trouver la meilleure valeur"
        )
        
        if knn_mode == "Un seul k":
            classifier_params["k"] = st.sidebar.slider("k (nombre de voisins)", 1, 20, 5)
            classifier_params["evaluate_range"] = False
        else:
            classifier_params["k_range"] = list(KNN_K_RANGE)
            classifier_params["evaluate_range"] = True
            st.sidebar.info("k sera évalué de 1 à 10")
        
        classifier_params["metric"] = st.sidebar.selectbox(
            "Métrique de distance",
            ("euclidean", "manhattan", "minkowski")
        )
        classifier_params["weights"] = st.sidebar.selectbox(
            "Pondération",
            ("uniform", "distance"),
            help="'uniform': tous les voisins égaux, 'distance': pondéré par l'inverse de la distance"
        )
        
    elif classifier_choice == "Naive Bayes":
        st.sidebar.markdown("**Naive Bayes** : Classifieur probabiliste bayésien")
        
        classifier_params["type"] = st.sidebar.selectbox(
            "Type de Naive Bayes",
            ("gaussian", "multinomial", "bernoulli"),
            help="Gaussian: pour features continues, Multinomial: pour comptages, Bernoulli: pour binaires"
        )
        
        if classifier_params["type"] == "gaussian":
            classifier_params["var_smoothing"] = st.sidebar.slider(
                "var_smoothing",
                min_value=1e-12,
                max_value=1e-6,
                value=1e-9,
                format="%.1e",
                help="Portion de la plus grande variance ajoutée pour la stabilité"
            )
        else:
            classifier_params["alpha"] = st.sidebar.slider(
                "alpha (lissage Laplace)",
                min_value=0.0,
                max_value=2.0,
                value=1.0,
                step=0.1
            )
        
    elif classifier_choice == "C4.5":
        st.sidebar.markdown("**C4.5** : Arbre de décision (gain d'information)")
        
        classifier_params["criterion"] = st.sidebar.selectbox(
            "Critère de division",
            ("entropy", "gini"),
            help="'entropy' correspond au gain d'information (C4.5), 'gini' est utilisé par CART"
        )
        
        max_depth_enabled = st.sidebar.checkbox("Limiter la profondeur", value=False)
        if max_depth_enabled:
            classifier_params["max_depth"] = st.sidebar.slider("Profondeur maximale", 1, 20, 5)
        else:
            classifier_params["max_depth"] = None
        
        classifier_params["min_samples_split"] = st.sidebar.slider(
            "min_samples_split",
            min_value=2,
            max_value=20,
            value=2,
            help="Nombre minimum d'échantillons pour diviser un nœud"
        )
        
        classifier_params["min_samples_leaf"] = st.sidebar.slider(
            "min_samples_leaf",
            min_value=1,
            max_value=10,
            value=1,
            help="Nombre minimum d'échantillons dans une feuille"
        )
        
    elif classifier_choice == "SVM":
        st.sidebar.markdown("**SVM** : Machine à vecteurs de support")
        
        classifier_params["kernel"] = st.sidebar.selectbox(
            "Type de noyau (kernel)",
            ("rbf", "linear", "poly", "sigmoid"),
            help="RBF est le plus utilisé pour les données non-linéaires"
        )
        
        classifier_params["C"] = st.sidebar.slider(
            "C (régularisation)",
            min_value=0.01,
            max_value=100.0,
            value=1.0,
            help="Plus C est grand, moins de tolérance aux erreurs"
        )
        
        if classifier_params["kernel"] == "rbf":
            gamma_option = st.sidebar.selectbox(
                "gamma",
                ("scale", "auto", "manuel"),
                help="Coefficient du noyau RBF"
            )
            if gamma_option == "manuel":
                classifier_params["gamma"] = st.sidebar.slider("Valeur de gamma", 0.001, 10.0, 1.0)
            else:
                classifier_params["gamma"] = gamma_option
        
        if classifier_params["kernel"] == "poly":
            classifier_params["degree"] = st.sidebar.slider("Degré polynomial", 2, 5, 3)
        
        classifier_params["normalize"] = st.sidebar.checkbox(
            "Normaliser les features",
            value=True,
            help="Recommandé pour SVM"
        )
    
    # Action buttons
    st.sidebar.markdown("---")
    col1, col2 = st.sidebar.columns(2)
    with col1:
        run_button = st.button("▶️ Classifier", use_container_width=True, key="btn_classify")
    with col2:
        compare_button = st.button("Comparer tous", use_container_width=True, key="btn_compare_all")
    
    return classifier_choice, classifier_params, run_button, compare_button


def render_classification_footer():
    """Render classification help in sidebar footer."""
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Aide Classification")
    st.sidebar.markdown("""
**Métriques :**
- **Précision** : TP / (TP + FP)
- **Rappel** : TP / (TP + FN)
- **F-mesure** : 2 × (P × R) / (P + R)

**Matrice de confusion:**
- **TP** : Vrais positifs
- **TN** : Vrais négatifs
- **FP** : Faux positifs
- **FN** : Faux négatifs
""")
