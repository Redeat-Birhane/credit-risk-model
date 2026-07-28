import pandas as pd
import numpy as np
from typing import List, Tuple
from src.config import DataConfig

def load_and_clean_data(config: DataConfig) -> pd.DataFrame:
    """Loads raw data and performs initial cleaning."""
    # Implementation for loading and cleaning
    pass

def compute_rfms_features(df: pd.DataFrame, config: DataConfig) -> pd.DataFrame:
    """Computes Recency, Frequency, Monetary, and Std features."""
    # Implementation for RFMS calculation
    pass

def split_features_target(df: pd.DataFrame, target_col: str) -> Tuple[pd.DataFrame, pd.Series]:
    """Splits the dataframe into features and target."""
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y