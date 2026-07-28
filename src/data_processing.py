"""
Data processing module for credit risk modeling.
Handles loading, cleaning, and feature engineering.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional, Dict, List, Union
from pathlib import Path


def load_data(file_path: Union[str, Path]) -> pd.DataFrame:
    """
    Load credit risk dataset from CSV file.
    
    Args:
        file_path: Path to the CSV file
        
    Returns:
        pd.DataFrame: Loaded dataset
        
    Raises:
        FileNotFoundError: If the file doesn't exist
        pd.errors.EmptyDataError: If the file is empty
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {file_path}")
    
    return pd.read_csv(file_path)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform initial data cleaning operations.
    
    Args:
        df: Raw dataframe
        
    Returns:
        pd.DataFrame: Cleaned dataframe with no missing values
    """
    # Remove duplicates
    df = df.drop_duplicates()
    
    # Handle missing values
    df = df.fillna(df.median(numeric_only=True))
    
    return df


def engineer_features(
    df: pd.DataFrame,
    categorical_columns: Optional[List[str]] = None,
    scale_numeric: bool = True
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Engineer features for credit risk modeling.
    
    Args:
        df: Cleaned dataframe
        categorical_columns: List of categorical column names to encode
        scale_numeric: Whether to standardize numeric features
        
    Returns:
        Tuple[pd.DataFrame, pd.Series]: Feature matrix and target variable
        
    Raises:
        ValueError: If target column 'default' is not found
    """
    # Ensure target exists
    if 'default' not in df.columns:
        raise ValueError("Target column 'default' not found in dataset")
    
    # Separate features and target
    X = df.drop(columns=['default'])
    y = df['default']
    
    # Encode categorical variables if specified
    if categorical_columns:
        X = pd.get_dummies(X, columns=categorical_columns, drop_first=True)
    
    # Scale numeric features (optional)
    if scale_numeric:
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        X[numeric_cols] = scaler.fit_transform(X[numeric_cols])
    
    return X, y


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42
) -> Dict[str, Union[pd.DataFrame, pd.Series]]:
    """
    Split data into train and test sets.
    
    Args:
        X: Feature matrix
        y: Target variable
        test_size: Proportion of data for testing (default: 0.2)
        random_state: Random seed for reproducibility
        
    Returns:
        Dict containing X_train, X_test, y_train, y_test
    """
    from sklearn.model_selection import train_test_split
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    return {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test
    }


def calculate_woe_iv(
    df: pd.DataFrame,
    feature: str,
    target: str = 'default',
    bins: int = 10
) -> Tuple[pd.DataFrame, float, float]:
    """
    Calculate Weight of Evidence (WoE) and Information Value (IV).
    
    Args:
        df: DataFrame containing feature and target
        feature: Feature column name
        target: Target column name (binary, 0/1)
        bins: Number of bins for continuous features
        
    Returns:
        Tuple[pd.DataFrame, float, float]: 
            - DataFrame with bin statistics
            - WoE values for each bin
            - Information Value
        
    Raises:
        ValueError: If target values are not 0 or 1
    """
    # Validate target
    if not set(df[target].unique()).issubset({0, 1}):
        raise ValueError("Target must be binary (0 and 1)")
    
    # Create bins
    df_binned = df.copy()
    if df[feature].dtype in ['int64', 'float64']:
        df_binned['bin'] = pd.qcut(df[feature], q=bins, duplicates='drop')
    else:
        df_binned['bin'] = df[feature]
    
    # Calculate WoE and IV
    grouped = df_binned.groupby('bin')[target].agg(['count', 'sum'])
    grouped['non_events'] = grouped['count'] - grouped['sum']
    
    total_events = grouped['sum'].sum()
    total_non_events = grouped['non_events'].sum()
    
    grouped['dist_events'] = grouped['sum'] / total_events
    grouped['dist_non_events'] = grouped['non_events'] / total_non_events
    
    # WoE calculation
    grouped['woe'] = np.log(
        grouped['dist_events'] / grouped['dist_non_events']
    ).replace([np.inf, -np.inf], 0)
    
    # IV calculation
    grouped['iv'] = (
        grouped['dist_events'] - grouped['dist_non_events']
    ) * grouped['woe']
    
    iv = grouped['iv'].sum()
    
    return grouped, grouped['woe'].values, iv


# Example of dataclass for configuration
from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelConfig:
    """Configuration for credit risk model."""
    
    # Data parameters
    data_path: str = "data/raw/credit_data.csv"
    test_size: float = 0.2
    random_state: int = 42
    
    # Model parameters
    model_type: str = "logistic_regression"  # or "gradient_boosting"
    max_iter: int = 1000
    regularization: str = "l2"
    C: float = 1.0
    
    # Training parameters
    cv_folds: int = 5
    scoring_metric: str = "roc_auc"
    
    # MLflow tracking
    experiment_name: str = "credit_risk_model"
    run_name: Optional[str] = None