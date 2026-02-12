"""Data preprocessing utilities."""

import logging
from typing import Optional, Tuple, List
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataPreprocessor:
    """Data preprocessing pipeline."""
    
    def __init__(self, scaling_method: str = "standard"):
        """
        Initialize preprocessor.
        
        Args:
            scaling_method: Scaling method ('standard', 'minmax', or 'none')
        """
        self.scaling_method = scaling_method
        self.scaler = None
        self.label_encoder = None
        self.feature_names = None
        
        if scaling_method == "standard":
            self.scaler = StandardScaler()
        elif scaling_method == "minmax":
            self.scaler = MinMaxScaler()
        elif scaling_method != "none":
            raise ValueError(f"Unknown scaling method: {scaling_method}")
    
    def fit_transform(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Fit preprocessor and transform data.
        
        Args:
            X: Features DataFrame
            y: Optional target Series
            
        Returns:
            Tuple of (transformed X, transformed y)
        """
        self.feature_names = X.columns.tolist()
        
        # Scale features
        if self.scaler is not None:
            X_scaled = self.scaler.fit_transform(X)
            logger.info(f"Fitted {self.scaling_method} scaler on {X.shape[1]} features")
        else:
            X_scaled = X.values
        
        # Encode labels if provided
        y_encoded = None
        if y is not None:
            if y.dtype == 'object' or isinstance(y.iloc[0], str):
                self.label_encoder = LabelEncoder()
                y_encoded = self.label_encoder.fit_transform(y)
                logger.info(f"Encoded labels: {len(self.label_encoder.classes_)} classes")
            else:
                y_encoded = y.values
        
        return X_scaled, y_encoded
    
    def transform(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Transform data using fitted preprocessor.
        
        Args:
            X: Features DataFrame
            y: Optional target Series
            
        Returns:
            Tuple of (transformed X, transformed y)
        """
        # Scale features
        if self.scaler is not None:
            X_scaled = self.scaler.transform(X)
        else:
            X_scaled = X.values
        
        # Encode labels if provided
        y_encoded = None
        if y is not None:
            if self.label_encoder is not None:
                y_encoded = self.label_encoder.transform(y)
            else:
                y_encoded = y.values
        
        return X_scaled, y_encoded
    
    def inverse_transform_labels(self, y: np.ndarray) -> np.ndarray:
        """Inverse transform labels back to original values."""
        if self.label_encoder is not None:
            return self.label_encoder.inverse_transform(y)
        return y


def remove_outliers(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    n_std: float = 3.0
) -> pd.DataFrame:
    """
    Remove outliers using standard deviation method.
    
    Args:
        df: Input DataFrame
        columns: Columns to check for outliers (None = all numeric)
        n_std: Number of standard deviations for threshold
        
    Returns:
        DataFrame with outliers removed
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    mask = pd.Series([True] * len(df))
    
    for col in columns:
        mean = df[col].mean()
        std = df[col].std()
        mask &= (df[col] >= mean - n_std * std) & (df[col] <= mean + n_std * std)
    
    n_removed = len(df) - mask.sum()
    logger.info(f"Removed {n_removed} outliers ({n_removed/len(df)*100:.2f}%)")
    
    return df[mask]


def handle_missing_values(
    df: pd.DataFrame,
    strategy: str = "mean",
    fill_value: Optional[float] = None
) -> pd.DataFrame:
    """
    Handle missing values in DataFrame.
    
    Args:
        df: Input DataFrame
        strategy: Strategy for handling missing values ('mean', 'median', 'mode', 'drop', 'fill')
        fill_value: Value to fill if strategy is 'fill'
        
    Returns:
        DataFrame with missing values handled
    """
    if df.isnull().sum().sum() == 0:
        logger.info("No missing values found")
        return df
    
    df = df.copy()
    n_missing = df.isnull().sum().sum()
    
    if strategy == "drop":
        df = df.dropna()
        logger.info(f"Dropped {n_missing} rows with missing values")
    elif strategy == "mean":
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
        logger.info(f"Filled missing values with mean")
    elif strategy == "median":
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
        logger.info(f"Filled missing values with median")
    elif strategy == "mode":
        for col in df.columns:
            df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 0)
        logger.info(f"Filled missing values with mode")
    elif strategy == "fill":
        if fill_value is None:
            raise ValueError("fill_value must be provided when strategy is 'fill'")
        df = df.fillna(fill_value)
        logger.info(f"Filled missing values with {fill_value}")
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    
    return df


def create_feature_crosses(
    df: pd.DataFrame,
    feature_pairs: List[Tuple[str, str]]
) -> pd.DataFrame:
    """
    Create interaction features (feature crosses).
    
    Args:
        df: Input DataFrame
        feature_pairs: List of feature pairs to cross
        
    Returns:
        DataFrame with additional crossed features
    """
    df = df.copy()
    
    for feat1, feat2 in feature_pairs:
        cross_name = f"{feat1}_x_{feat2}"
        df[cross_name] = df[feat1] * df[feat2]
        logger.debug(f"Created feature cross: {cross_name}")
    
    logger.info(f"Created {len(feature_pairs)} feature crosses")
    return df


def reduce_memory_usage(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Reduce memory usage by downcasting numeric types.
    
    Args:
        df: Input DataFrame
        verbose: Whether to print memory savings
        
    Returns:
        DataFrame with reduced memory usage
    """
    start_mem = df.memory_usage().sum() / 1024**2
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    
    end_mem = df.memory_usage().sum() / 1024**2
    
    if verbose:
        logger.info(f'Memory usage decreased from {start_mem:.2f} MB to {end_mem:.2f} MB '
                   f'({100 * (start_mem - end_mem) / start_mem:.1f}% reduction)')
    
    return df
