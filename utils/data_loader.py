"""
Data loading and validation utilities.
"""

import numpy as np
import pandas as pd
import streamlit as st


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
