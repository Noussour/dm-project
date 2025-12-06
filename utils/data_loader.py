"""
Data loading and validation utilities.
"""

import os
import numpy as np
import pandas as pd
import streamlit as st

from config.constants import DATASETS_DIR, PREDEFINED_DATASETS, ALGORITHM_CONSTRAINTS


def get_predefined_datasets() -> dict:
    """
    Get list of available predefined datasets.
    
    Returns:
        Dictionary of dataset info with keys being dataset names
    """
    return PREDEFINED_DATASETS


def load_predefined_dataset(dataset_name: str) -> tuple[pd.DataFrame | None, str | None]:
    """
    Load a predefined dataset from the datasets directory.
    
    Args:
        dataset_name: Name of the dataset (key in PREDEFINED_DATASETS)
        
    Returns:
        Tuple of (DataFrame, filename) or (None, None) if loading fails
    """
    if dataset_name not in PREDEFINED_DATASETS:
        st.error(f"Dataset inconnu: {dataset_name}")
        return None, None
    
    dataset_info = PREDEFINED_DATASETS[dataset_name]
    filepath = os.path.join(DATASETS_DIR, dataset_info["filename"])
    
    if not os.path.exists(filepath):
        st.error(f"Fichier introuvable: {dataset_info['filename']}")
        return None, None
    
    try:
        # Define missing value indicators
        missing_values = ['?', 'NA', 'N/A', 'null', 'None', 'MISSING', '-', '']
        
        df = pd.read_csv(filepath, na_values=missing_values, keep_default_na=True)
        
        # Handle datasets with unnamed columns (like ecoli)
        # Rename unnamed columns
        new_columns = []
        for i, col in enumerate(df.columns):
            if col.startswith('Unnamed') or col == '':
                new_columns.append(f'feature_{i}')
            else:
                new_columns.append(col)
        df.columns = new_columns
        
        return df, dataset_info["filename"]
    except Exception as e:
        st.error(f"Erreur lors du chargement du dataset: {e}")
        return None, None


def validate_algorithm_compatibility(df: pd.DataFrame, selected_features: list, 
                                     algo_name: str, target_col: str = None) -> tuple[bool, list[str]]:
    """
    Validate if the dataset is compatible with the selected algorithm.
    
    Args:
        df: DataFrame with data
        selected_features: List of selected feature columns
        algo_name: Name of the algorithm
        target_col: Target column for classification (optional)
        
    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    errors = []
    warnings = []
    
    if algo_name not in ALGORITHM_CONSTRAINTS:
        return True, []
    
    constraints = ALGORITHM_CONSTRAINTS[algo_name]
    
    # Get data subset
    if selected_features:
        X = df[selected_features]
    else:
        X = df.select_dtypes(include=[np.number])
    
    # Check minimum samples
    n_samples = len(df)
    if n_samples < constraints.get("min_samples", 1):
        errors.append(
            f"Nombre d'échantillons insuffisant: {n_samples} < {constraints['min_samples']} requis pour {algo_name}"
        )
    
    # Check minimum features
    n_features = len(selected_features) if selected_features else X.shape[1]
    if n_features < constraints.get("min_features", 1):
        errors.append(
            f"Nombre de features insuffisant: {n_features} < {constraints['min_features']} requis pour {algo_name}"
        )
    
    # Check for missing values
    if constraints.get("requires_no_missing", False):
        missing_count = X.isnull().sum().sum()
        if missing_count > 0:
            errors.append(
                f"Le dataset contient {missing_count} valeur(s) manquante(s). "
                f"{algo_name} nécessite des données complètes. "
                "Utilisez le prétraitement pour gérer les valeurs manquantes."
            )
    
    # Check for non-negative values (Multinomial Naive Bayes)
    if constraints.get("requires_non_negative", False):
        if (X < 0).any().any():
            warnings.append(
                f"Certaines valeurs sont négatives. Le Naive Bayes Multinomial "
                "ne fonctionne qu'avec des valeurs non-négatives. "
                "Utilisez le type Gaussian ou normalisez vos données."
            )
    
    # Check minimum classes for classification
    if target_col and "min_classes" in constraints:
        if target_col in df.columns:
            n_classes = df[target_col].nunique()
            if n_classes < constraints["min_classes"]:
                errors.append(
                    f"Nombre de classes insuffisant: {n_classes} < {constraints['min_classes']} requis pour {algo_name}"
                )
            
            # Check minimum samples per class for stratification
            class_counts = df[target_col].value_counts()
            min_class_samples = class_counts.min()
            if min_class_samples < 2:
                warnings.append(
                    f"La classe la moins représentée n'a que {min_class_samples} échantillon(s). "
                    "La stratification sera désactivée."
                )
    
    # Special validation for n_clusters parameter
    if algo_name in ["KMeans", "K-Medoids", "AGNES", "DIANA"]:
        max_clusters = n_samples - 1
        if max_clusters < 2:
            errors.append(
                f"Trop peu d'échantillons pour le clustering: minimum 3 échantillons nécessaires."
            )
    
    return len(errors) == 0, errors + warnings


def validate_visualization_compatibility(df: pd.DataFrame, selected_features: list,
                                         algo_name: str, viz_type: str) -> tuple[bool, str]:
    """
    Validate if visualization is compatible with the data and algorithm.
    
    Args:
        df: DataFrame with data
        selected_features: List of selected feature columns
        algo_name: Name of the algorithm
        viz_type: Type of visualization ('2D_scatter', '3D_scatter', 'dendrogram', etc.)
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    from config.constants import VISUALIZATION_CONSTRAINTS
    
    if viz_type not in VISUALIZATION_CONSTRAINTS:
        return True, ""
    
    constraints = VISUALIZATION_CONSTRAINTS[viz_type]
    n_features = len(selected_features)
    n_samples = len(df)
    
    # Check minimum features
    if "min_features" in constraints:
        if n_features < constraints["min_features"]:
            return False, (
                f"{constraints['description']}. "
                f"Vous n'avez que {n_features} feature(s) sélectionnée(s)."
            )
    
    # Check maximum samples (for dendrogram)
    if "max_samples" in constraints:
        if n_samples > constraints["max_samples"]:
            return False, (
                f"Trop d'échantillons pour {viz_type}: {n_samples} > {constraints['max_samples']} max. "
                "Utilisez un sous-échantillonnage ou une autre visualisation."
            )
    
    # Check algorithm compatibility
    if "algorithms" in constraints:
        if algo_name not in constraints["algorithms"]:
            return False, (
                f"{constraints['description']}. "
                f"L'algorithme actuel est {algo_name}."
            )
    
    return True, ""


def validate_clustering_params(algo_name: str, params: dict, n_samples: int) -> tuple[bool, str]:
    """
    Validate clustering algorithm parameters against data constraints.
    
    Args:
        algo_name: Name of the algorithm
        params: Algorithm parameters dictionary
        n_samples: Number of samples in the dataset
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if algo_name in ["KMeans", "K-Medoids", "AGNES", "DIANA"]:
        n_clusters = params.get("n_clusters", 2)
        if n_clusters >= n_samples:
            return False, (
                f"Le nombre de clusters ({n_clusters}) doit être inférieur "
                f"au nombre d'échantillons ({n_samples})."
            )
        if n_clusters < 2:
            return False, "Le nombre de clusters doit être au moins 2."
    
    if algo_name == "DBSCAN":
        min_samples = params.get("min_samples", 5)
        if min_samples >= n_samples:
            return False, (
                f"Le paramètre min_samples ({min_samples}) doit être inférieur "
                f"au nombre d'échantillons ({n_samples})."
            )
        eps = params.get("eps", 0.5)
        if eps <= 0:
            return False, "Le paramètre eps doit être positif."
    
    return True, ""


@st.cache_data(show_spinner=False)
def read_uploaded_file(uploaded_file) -> pd.DataFrame | None:
    """
    Read a CSV or Excel file and return a DataFrame.
    Automatically treats "?" as missing values.
    
    Args:
        uploaded_file: Streamlit uploaded file object
        
    Returns:
        pd.DataFrame or None if reading fails
    """
    if uploaded_file is None:
        return None
    
    name = uploaded_file.name.lower()
    
    try:
        # Define missing value indicators
        missing_values = ['?', 'NA', 'N/A', 'null', 'None', 'MISSING', '-', '']
        
        if name.endswith(".csv"):
            df = pd.read_csv(uploaded_file, na_values=missing_values, keep_default_na=True)
        elif name.endswith((".xls", ".xlsx")):
            df = pd.read_excel(uploaded_file, na_values=missing_values, keep_default_na=True)
        else:
            st.error("Format non supporté. Uploadez un .csv ou .xlsx/.xls.")
            return None
        
        # Also convert empty strings to NaN for CSV files
        if name.endswith(".csv"):
            df = df.replace(r'^\s*$', np.nan, regex=True)
        
        return df
    except Exception as e:
        st.error(f"Erreur lors de la lecture du fichier: {e}")
        return None


def validate_dataframe(df: pd.DataFrame) -> tuple[bool, str]:
    """
    Validate a DataFrame for clustering.
    Missing values are allowed - users can clean them in the preprocessing section.
    
    Args:
        df: DataFrame to validate
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if df is None:
        return False, "Veuillez uploader un dataset (CSV ou Excel) via la sidebar."
    
    # Check for numeric columns (at least 1 numeric column required)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) < 1:
        return False, "Le dataset doit contenir au moins 1 feature numérique pour le clustering."
    
    return True, ""


def get_numeric_columns(df: pd.DataFrame) -> list[str]:
    """
    Get list of numeric column names from DataFrame.
    
    Args:
        df: Input DataFrame
        
    Returns:
        List of numeric column names
    """
    return df.select_dtypes(include=[np.number]).columns.tolist()


def compute_five_number_summary(df: pd.DataFrame) -> pd.DataFrame | None:
    """
    Compute five-number summary for numeric columns.
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with five-number summary or None if no numeric columns
    """
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(num_cols) == 0:
        return None
    
    five_num = pd.DataFrame(
        index=num_cols,
        columns=["Minimum", "Q1", "Médiane", "Q3", "Maximum"]
    )
    
    for col in num_cols:
        try:
            five_num.loc[col, "Minimum"] = df[col].min()
            five_num.loc[col, "Q1"] = df[col].quantile(0.25)
            five_num.loc[col, "Médiane"] = df[col].median()
            five_num.loc[col, "Q3"] = df[col].quantile(0.75)
            five_num.loc[col, "Maximum"] = df[col].max()
        except Exception:
            five_num.loc[col, :] = [None] * 5
    
    try:
        five_num = five_num.astype(float).round(4)
    except Exception:
        pass
    
    return five_num


def filter_dataframe(df: pd.DataFrame, filters: dict) -> pd.DataFrame:
    """
    Apply filters to a DataFrame.
    
    Args:
        df: Input DataFrame
        filters: Dictionary with column names as keys and (min, max) tuples as values
        
    Returns:
        Filtered DataFrame
    """
    filtered_df = df.copy()
    
    for col, (min_val, max_val) in filters.items():
        if col in filtered_df.columns:
            filtered_df = filtered_df[
                (filtered_df[col] >= min_val) & (filtered_df[col] <= max_val)
            ]
    
    return filtered_df
