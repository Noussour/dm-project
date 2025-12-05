"""
Advanced preprocessing UI components with outlier detection, boxplots, and strategy management.
"""

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from utils.preprocessing import PreprocessingPipeline, MissingDataHandler, OutlierHandler, DataTransformer


def render_outlier_detection_section(df: pd.DataFrame, selected_features: list) -> tuple[bool, list]:
    """
    Render outlier detection with boxplots and removal checkbox.
    
    Args:
        df: Input DataFrame
        selected_features: Selected numeric feature columns
        
    Returns:
        Tuple of (remove_outliers_checkbox, outlier_features_to_remove)
    """
    st.markdown("#### **Détection et suppression des valeurs aberrantes (Outliers)**")
    
    # Checkbox to enable outlier removal
    remove_outliers = st.checkbox(
        "Détecter et supprimer les valeurs aberrantes",
        value=False,
        key="remove_outliers_checkbox"
    )
    
    if remove_outliers and selected_features:
        st.markdown("_Boxplots des features sélectionnées (les points en dehors indiquent des valeurs aberrantes)_")
        
        # Detect outliers
        outlier_handler = OutlierHandler()
        outliers_info = outlier_handler.detect_outliers_iqr(df, columns=selected_features)
        
        # Create boxplots
        col1, col2 = st.columns(2)
        
        with col1:
            # Create boxplot figure
            fig = go.Figure()
            for feature in selected_features:
                fig.add_trace(go.Box(
                    y=df[feature].dropna(),
                    name=feature,
                    boxmean='sd'
                ))
            
            fig.update_layout(
                title="Boxplots des Features",
                yaxis_title="Valeur",
                height=400,
                showlegend=True
            )
            st.plotly_chart(fig, width='stretch')
        
        with col2:
            # Show outlier statistics
            st.markdown("**Statistiques des outliers (IQR):**")
            for feature, info in outliers_info.items():
                if info['count'] > 0:
                    st.warning(
                        f"**{feature}**: {info['count']} outliers ({info['percentage']:.1f}%)\n"
                        f"Plage: [{info['lower_bound']:.2f}, {info['upper_bound']:.2f}]"
                    )
                else:
                    st.success(f"**{feature}**: Aucun outlier détecté ✓")
        
        # Select which features to remove outliers from
        outlier_features = st.multiselect(
            "Sélectionner les features pour lesquelles supprimer les outliers",
            [f for f, info in outliers_info.items() if info['count'] > 0],
            default=[f for f, info in outliers_info.items() if info['count'] > 0],
            key="outlier_features_select"
        )
        
        return remove_outliers, outlier_features
    
    return remove_outliers, []


def render_missing_values_strategy(df: pd.DataFrame) -> dict:
    """
    Render strategy selection for missing values with global and per-column options.
    
    Args:
        df: Input DataFrame
        
    Returns:
        Dictionary with strategy configuration
    """
    st.markdown("#### **Gestion des valeurs manquantes**")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Global strategy
    st.markdown("**Stratégie globale (par défaut pour toutes les colonnes):**")
    global_strategy = st.selectbox(
        "Sélectionnez une stratégie",
        [
            "Aucune (garder les valeurs manquantes)",
            "Supprimer les lignes avec valeurs manquantes",
            "Supprimer les colonnes avec valeurs manquantes",
            "Imputation par la moyenne",
            "Imputation par la médiane",
            "Forward Fill (dernière observation)",
            "Backward Fill (observation suivante)",
        ],
        index=3,
        key="global_strategy",
        label_visibility="collapsed"
    )
    
    # Per-column specific strategies
    per_column_strategies = {}
    if df.isnull().any().any():
        st.markdown("**Stratégies spécifiques par colonne (optionnel):**")
        
        cols_with_missing = df.columns[df.isnull().any()].tolist()
        
        with st.expander("Définir des stratégies spécifiques pour certaines colonnes"):
            for col in cols_with_missing:
                if col in numeric_cols:
                    col_strategy = st.selectbox(
                        f"Stratégie pour **{col}**:",
                        [
                            "Utiliser la stratégie globale",
                            "Imputation par la moyenne",
                            "Imputation par la médiane",
                            "Forward Fill",
                            "Backward Fill",
                            "Supprimer cette colonne",
                        ],
                        index=0,
                        key=f"col_strategy_{col}"
                    )
                    if col_strategy != "Utiliser la stratégie globale":
                        per_column_strategies[col] = col_strategy
    
    return {
        "global_strategy": global_strategy,
        "per_column_strategies": per_column_strategies
    }


def render_normalization_section(df: pd.DataFrame) -> dict:
    """
    Render data normalization/transformation options.
    
    Args:
        df: Input DataFrame
        
    Returns:
        Dictionary with normalization configuration
    """
    st.markdown("#### **Normalisation des features numériques**")
    
    normalization_option = st.selectbox(
        "Sélectionner une stratégie de normalisation",
        [
            "Aucune",
            "Min-Max (mise à l'échelle 0-1)",
            "Z-Score (standardisation)",
            "Decimal Scaling",
        ],
        key="normalization_option"
    )
    
    config = {"strategy": normalization_option}
    
    if normalization_option == "Min-Max (mise à l'échelle 0-1)":
        col1, col2 = st.columns(2)
        with col1:
            config["min_value"] = st.number_input("Valeur min", value=0.0, key="minmax_min")
        with col2:
            config["max_value"] = st.number_input("Valeur max", value=1.0, key="minmax_max")
    
    return config


def run_preprocessing_pipeline(
    df: pd.DataFrame,
    selected_features: list,
    remove_outliers: bool,
    outlier_features: list,
    missing_strategy: dict,
    normalization_config: dict
) -> pd.DataFrame:
    """
    Execute the complete preprocessing pipeline.
    
    Args:
        df: Input DataFrame
        selected_features: Selected numeric features
        remove_outliers: Whether to remove outliers
        outlier_features: Features to remove outliers from
        missing_strategy: Missing values strategy config
        normalization_config: Normalization strategy config
        
    Returns:
        Preprocessed DataFrame
    """
    df_processed = df.copy()
    steps_log = []
    
    # Step 1: Remove outliers
    if remove_outliers and outlier_features:
        outlier_handler = OutlierHandler()
        df_processed, removed_count = outlier_handler.remove_outliers_iqr(df_processed, columns=outlier_features)
        steps_log.append(f"✓ Suppression des outliers: {removed_count} lignes supprimées")
    
    # Step 2: Handle missing values
    global_strat = missing_strategy["global_strategy"]
    per_col_strats = missing_strategy["per_column_strategies"]
    
    if global_strat != "Aucune (garder les valeurs manquantes)":
        handler = MissingDataHandler()
        
        # Apply per-column strategies first
        for col, col_strat in per_col_strats.items():
            if col_strat == "Imputation par la moyenne":
                df_processed[col] = handler.impute_mean(df_processed[[col]])
                steps_log.append(f"✓ {col}: imputation par la moyenne")
            elif col_strat == "Imputation par la médiane":
                df_processed[col] = handler.impute_median(df_processed[[col]])
                steps_log.append(f"✓ {col}: imputation par la médiane")
            elif col_strat == "Forward Fill":
                df_processed[col] = handler.impute_forward_fill(df_processed[[col]])
                steps_log.append(f"✓ {col}: forward fill")
            elif col_strat == "Backward Fill":
                df_processed[col] = handler.impute_backward_fill(df_processed[[col]])
                steps_log.append(f"✓ {col}: backward fill")
            elif col_strat == "Supprimer cette colonne":
                df_processed = df_processed.drop(columns=[col])
                steps_log.append(f"✓ {col}: colonne supprimée")
        
        # Apply global strategy to remaining missing values
        if df_processed.isnull().values.any():
            if global_strat == "Supprimer les lignes avec valeurs manquantes":
                initial_len = len(df_processed)
                df_processed = df_processed.dropna()
                removed = initial_len - len(df_processed)
                steps_log.append(f"✓ Suppression des lignes: {removed} lignes supprimées")
            
            elif global_strat == "Supprimer les colonnes avec valeurs manquantes":
                initial_cols = len(df_processed.columns)
                df_processed = df_processed.dropna(axis=1)
                removed_cols = initial_cols - len(df_processed.columns)
                steps_log.append(f"✓ Suppression des colonnes: {removed_cols} colonnes supprimées")
            
            elif global_strat == "Imputation par la moyenne":
                df_processed = handler.impute_mean(df_processed)
                steps_log.append("✓ Imputation par la moyenne (globale)")
            
            elif global_strat == "Imputation par la médiane":
                df_processed = handler.impute_median(df_processed)
                steps_log.append("✓ Imputation par la médiane (globale)")
            
            elif global_strat == "Forward Fill (dernière observation)":
                df_processed = handler.impute_forward_fill(df_processed)
                steps_log.append("✓ Forward Fill (globale)")
            
            elif global_strat == "Backward Fill (observation suivante)":
                df_processed = handler.impute_backward_fill(df_processed)
                steps_log.append("✓ Backward Fill (globale)")
    
    # Step 3: Apply normalization
    norm_strat = normalization_config["strategy"]
    if norm_strat != "Aucune":
        transformer = DataTransformer()
        numeric_cols = df_processed.select_dtypes(include=[np.number]).columns.tolist()
        
        if norm_strat == "Min-Max (mise à l'échelle 0-1)":
            min_val = normalization_config.get("min_value", 0.0)
            max_val = normalization_config.get("max_value", 1.0)
            df_processed[numeric_cols] = transformer.apply_minmax_normalization(
                df_processed[numeric_cols], feature_range=(min_val, max_val)
            )
            steps_log.append(f"✓ Min-Max normalization: [{min_val}, {max_val}]")
        
        elif norm_strat == "Z-Score (standardisation)":
            df_processed[numeric_cols] = transformer.apply_zscore_normalization(df_processed[numeric_cols])
            steps_log.append("✓ Z-Score normalization (standardisation)")
        
        elif norm_strat == "Decimal Scaling":
            df_processed[numeric_cols] = transformer.apply_decimal_scaling(df_processed[numeric_cols])
            steps_log.append("✓ Decimal Scaling")
    
    # Display processing log
    st.success("Pipeline de prétraitement exécuté avec succès!")
    with st.expander("Détails des étapes exécutées"):
        for step in steps_log:
            st.write(step)
    
    # Show data shape change
    st.info(f"Dataset original: {df.shape} → Après prétraitement: {df_processed.shape}")
    
    return df_processed
