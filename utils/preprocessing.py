"""
Data preprocessing utilities for clustering.

This module provides functions for:
1. Data Cleaning: Handling missing data (row-level and cell-level)
2. Data Transformation: Normalization and discretization
3. Outlier Detection & Removal
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.impute import SimpleImputer


# ============================================================================
# UTILITY: REPLACE MISSING VALUE INDICATORS
# ============================================================================

def normalize_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Replace common missing value indicators with NaN.
    Treats "?", "NA", "N/A", "null", "None", "MISSING" as missing values.
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with normalized missing values (all converted to NaN)
    """
    df_clean = df.copy()
    missing_indicators = ['?', 'NA', 'N/A', 'null', 'None', 'MISSING', '-', '']
    df_clean = df_clean.replace(missing_indicators, np.nan)
    return df_clean


# ============================================================================
# OUTLIER DETECTION & REMOVAL
# ============================================================================

class OutlierHandler:
    """Handles detection and removal of outliers in datasets."""
    
    @staticmethod
    def detect_outliers_iqr(df: pd.DataFrame, columns: list = None) -> dict:
        """
        Detect outliers using Interquartile Range (IQR) method.
        
        Args:
            df: Input DataFrame
            columns: List of columns to check (numeric only). If None, checks all numeric columns.
            
        Returns:
            Dictionary with outlier information per column
        """
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        outliers_info = {}
        
        for col in columns:
            if col in df.columns and np.issubdtype(df[col].dtype, np.number):
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outlier_mask = (df[col] < lower_bound) | (df[col] > upper_bound)
                outlier_count = outlier_mask.sum()
                
                outliers_info[col] = {
                    'count': outlier_count,
                    'percentage': (outlier_count / len(df)) * 100,
                    'lower_bound': lower_bound,
                    'upper_bound': upper_bound,
                    'mask': outlier_mask
                }
        
        return outliers_info
    
    @staticmethod
    def remove_outliers_iqr(df: pd.DataFrame, columns: list = None) -> tuple[pd.DataFrame, int]:
        """
        Remove rows containing outliers (IQR method).
        
        Args:
            df: Input DataFrame
            columns: List of columns to check for outliers
            
        Returns:
            Tuple of (DataFrame without outliers, count of removed rows)
        """
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        df_clean = df.copy()
        initial_size = len(df_clean)
        
        for col in columns:
            if col in df_clean.columns and np.issubdtype(df_clean[col].dtype, np.number):
                Q1 = df_clean[col].quantile(0.25)
                Q3 = df_clean[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
        
        removed = initial_size - len(df_clean)
        return df_clean, removed


# ============================================================================
# DATA CLEANING STRATEGIES
# ============================================================================

class MissingDataHandler:
    """Handles missing data in datasets at both row and cell levels."""
    
    @staticmethod
    def ignore_records(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
        """
        Remove records (rows) that contain any missing values.
        ROW-LEVEL operation.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Tuple of (DataFrame with rows removed, count of removed rows)
        """
        # First normalize missing values
        df_normalized = normalize_missing_values(df)
        
        initial_size = len(df_normalized)
        df_clean = df_normalized.dropna()
        removed = initial_size - len(df_clean)
        return df_clean, removed
    
    @staticmethod
    def ignore_columns_with_missing(df: pd.DataFrame) -> tuple[pd.DataFrame, list]:
        """
        Remove entire columns (attributes) that contain any missing values.
        COLUMN-LEVEL operation.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Tuple of (DataFrame with columns removed, list of removed column names)
        """
        # First normalize missing values
        df_normalized = normalize_missing_values(df)
        df_clean = df_normalized.dropna(axis=1)
        removed_cols = [col for col in df.columns if col not in df_clean.columns]
        return df_clean, removed_cols
    
    @staticmethod
    def impute_mean(df: pd.DataFrame) -> pd.DataFrame:
        """
        Impute missing values (CELL-LEVEL) using column mean for numeric columns.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with missing values imputed by column mean
        """
        # First normalize missing values
        df_imputed = normalize_missing_values(df)
        numeric_cols = df_imputed.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if df_imputed[col].isnull().any():
                mean_val = df_imputed[col].mean()
                df_imputed[col].fillna(mean_val, inplace=True)
        
        return df_imputed
    
    @staticmethod
    def impute_median(df: pd.DataFrame) -> pd.DataFrame:
        """
        Impute missing values (CELL-LEVEL) using column median (robust to outliers).
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with missing values imputed by column median
        """
        # First normalize missing values
        df_imputed = normalize_missing_values(df)
        numeric_cols = df_imputed.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if df_imputed[col].isnull().any():
                median_val = df_imputed[col].median()
                df_imputed[col].fillna(median_val, inplace=True)
        
        return df_imputed
    
    @staticmethod
    def impute_forward_fill(df: pd.DataFrame) -> pd.DataFrame:
        """
        Impute missing values (CELL-LEVEL) using forward fill (last observation carried forward).
        Useful for time-series data.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with missing values imputed by forward fill
        """
        # First normalize missing values
        df_imputed = normalize_missing_values(df)
        df_imputed = df_imputed.fillna(method='ffill')
        # Backfill remaining NaN values (for first rows if they were NaN)
        df_imputed = df_imputed.fillna(method='bfill')
        return df_imputed
    
    @staticmethod
    def impute_backward_fill(df: pd.DataFrame) -> pd.DataFrame:
        """
        Impute missing values (CELL-LEVEL) using backward fill (next observation carried backward).
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with missing values imputed by backward fill
        """
        # First normalize missing values
        df_imputed = normalize_missing_values(df)
        df_imputed = df_imputed.fillna(method='bfill')
        # Forward fill remaining NaN values (for last rows if they were NaN)
        df_imputed = df_imputed.fillna(method='ffill')
    
    @staticmethod
    def impute_forward_fill(df: pd.DataFrame) -> pd.DataFrame:
        """
        Impute missing values (CELL-LEVEL) using forward fill (last observation carried forward).
        Useful for time-series data.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with missing values imputed by forward fill
        """
        df_imputed = df.fillna(method='ffill')
        # Backfill remaining NaN values (for first rows if they were NaN)
        df_imputed = df_imputed.fillna(method='bfill')
        return df_imputed
    
    @staticmethod
    def impute_backward_fill(df: pd.DataFrame) -> pd.DataFrame:
        """
        Impute missing values (CELL-LEVEL) using backward fill (next observation carried backward).
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with missing values imputed by backward fill
        """
        df_imputed = df.fillna(method='bfill')
        # Forward fill remaining NaN values (for last rows if they were NaN)
        df_imputed = df_imputed.fillna(method='ffill')
        return df_imputed


# ============================================================================
# DATA TRANSFORMATION & DISCRETIZATION STRATEGIES
# ============================================================================

class DataTransformer:
    """Handles data transformation and normalization."""
    
    @staticmethod
    def apply_minmax_normalization(df: pd.DataFrame, feature_range: tuple = (0, 1)) -> pd.DataFrame:
        """
        Apply Min-Max normalization to numeric columns.
        
        Scales values to range [new_min, new_max] using formula:
        X_scaled = (X - X_min) / (X_max - X_min) * (new_max - new_min) + new_min
        
        Args:
            df: Input DataFrame
            feature_range: Target range (default: (0, 1))
            
        Returns:
            DataFrame with Min-Max normalized numeric columns
        """
        df_normalized = df.copy()
        numeric_cols = df_normalized.select_dtypes(include=[np.number]).columns
        
        scaler = MinMaxScaler(feature_range=feature_range)
        df_normalized[numeric_cols] = scaler.fit_transform(df_normalized[numeric_cols])
        
        return df_normalized
    
    @staticmethod
    def apply_zscore_normalization(df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply Z-Score normalization (standardization) to numeric columns.
        
        Scales values based on mean and standard deviation:
        X_scaled = (X - mean) / std
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with Z-Score normalized numeric columns
        """
        df_normalized = df.copy()
        numeric_cols = df_normalized.select_dtypes(include=[np.number]).columns
        
        scaler = StandardScaler()
        df_normalized[numeric_cols] = scaler.fit_transform(df_normalized[numeric_cols])
        
        return df_normalized
    
    @staticmethod
    def apply_decimal_scaling(df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply decimal scaling to numeric columns.
        
        Moves the decimal point:
        X_scaled = X / 10^j, where j is the smallest integer such that max(|X_scaled|) < 1
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with decimal-scaled numeric columns
        """
        df_scaled = df.copy()
        numeric_cols = df_scaled.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            max_val = df_scaled[col].abs().max()
            if max_val > 0:
                j = len(str(int(max_val))) - 1
                divisor = 10 ** j
                df_scaled[col] = df_scaled[col] / divisor
        
        return df_scaled
    
    @staticmethod
    def apply_binning(df: pd.DataFrame, column: str, n_bins: int = 5, method: str = 'equal_width') -> pd.Series:
        """
        Apply binning/discretization to a numeric column.
        
        Args:
            df: Input DataFrame
            column: Column name to bin
            n_bins: Number of bins (default: 5)
            method: 'equal_width' or 'equal_frequency' (quantile-based)
            
        Returns:
            Series with binned values
        """
        if method == 'equal_width':
            binned = pd.cut(df[column], bins=n_bins, labels=False)
        else:  # equal_frequency (quantile-based)
            binned = pd.qcut(df[column], q=n_bins, labels=False, duplicates='drop')
        
        return binned


# ============================================================================
# PREPROCESSING PIPELINE
# ============================================================================

class PreprocessingPipeline:
    """Orchestrates multiple preprocessing steps."""
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize pipeline with a DataFrame.
        
        Args:
            df: Input DataFrame
        """
        self.df_original = df.copy()
        self.df_processed = df.copy()
        self.steps_applied = []
    
    def apply_missing_data_handling(self, strategy: str, target_col: str = None, knn_neighbors: int = 5) -> bool:
        """
        Apply missing data handling strategy (row-level or cell-level).
        
        Args:
            strategy: One of:
                - 'ignore_records' (ROW-LEVEL)
                - 'ignore_columns' (COLUMN-LEVEL)
                - 'impute_mean' (CELL-LEVEL)
                - 'impute_median' (CELL-LEVEL)
                - 'impute_forward_fill' (CELL-LEVEL)
                - 'impute_backward_fill' (CELL-LEVEL)
                - 'impute_knn' (CELL-LEVEL)
                - 'impute_class_mean' (CELL-LEVEL)
            target_col: Required for 'impute_class_mean'
            knn_neighbors: Number of neighbors for KNN imputation
            
        Returns:
            True if applied, False if failed or no missing data
        """
        # Check if any missing data exists
        if not self.df_processed.isnull().values.any():
            return False
        
        handler = MissingDataHandler()
        
        if strategy == 'ignore_records':
            self.df_processed, removed = handler.ignore_records(self.df_processed)
            self.steps_applied.append(f"Ignore records: {removed} rows removed")
            return True
        
        elif strategy == 'ignore_columns':
            self.df_processed, removed_cols = handler.ignore_columns_with_missing(self.df_processed)
            self.steps_applied.append(f"Ignore columns: {', '.join(removed_cols)} removed")
            return True
        
        elif strategy == 'impute_mean':
            self.df_processed = handler.impute_mean(self.df_processed)
            self.steps_applied.append("Impute missing with column mean (cell-level)")
            return True
        
        elif strategy == 'impute_median':
            self.df_processed = handler.impute_median(self.df_processed)
            self.steps_applied.append("Impute missing with column median (cell-level)")
            return True
        
        elif strategy == 'impute_forward_fill':
            self.df_processed = handler.impute_forward_fill(self.df_processed)
            self.steps_applied.append("Impute missing with forward fill (cell-level)")
            return True
        
        elif strategy == 'impute_backward_fill':
            self.df_processed = handler.impute_backward_fill(self.df_processed)
            self.steps_applied.append("Impute missing with backward fill (cell-level)")
            return True
        
        elif strategy == 'impute_knn':
            self.df_processed = handler.impute_knn(self.df_processed, n_neighbors=knn_neighbors)
            self.steps_applied.append(f"Impute missing with KNN (k={knn_neighbors}, cell-level)")
            return True
        
        elif strategy == 'impute_class_mean':
            result = handler.impute_class_mean(self.df_processed, target_col)
            if result is not None:
                self.df_processed = result
                self.steps_applied.append("Impute missing with class-specific mean")
                return True
            return False
        
        return False
    
    def apply_normalization(self, strategy: str, feature_range: tuple = (0, 1)) -> bool:
        """
        Apply normalization strategy.
        
        Args:
            strategy: One of 'minmax', 'zscore', 'decimal'
            feature_range: For minmax, the target range (default: (0, 1))
            
        Returns:
            True if applied
        """
        transformer = DataTransformer()
        
        if strategy == 'minmax':
            self.df_processed = transformer.apply_minmax_normalization(
                self.df_processed, feature_range=feature_range
            )
            self.steps_applied.append(f"Min-Max normalization (range: {feature_range})")
            return True
        
        elif strategy == 'zscore':
            self.df_processed = transformer.apply_zscore_normalization(self.df_processed)
            self.steps_applied.append("Z-Score normalization (standardization)")
            return True
        
        elif strategy == 'decimal':
            self.df_processed = transformer.apply_decimal_scaling(self.df_processed)
            self.steps_applied.append("Decimal scaling")
            return True
        
        return False
    
    def reset(self):
        """Reset to original DataFrame."""
        self.df_processed = self.df_original.copy()
        self.steps_applied = []
    
    def get_processed_data(self) -> pd.DataFrame:
        """Return the processed DataFrame."""
        return self.df_processed.copy()
    
    def get_steps_summary(self) -> str:
        """Return a summary of applied steps."""
        if not self.steps_applied:
            return "No preprocessing steps applied"
        return "\n".join([f"• {step}" for step in self.steps_applied])
