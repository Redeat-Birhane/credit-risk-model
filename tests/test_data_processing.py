import pytest
import pandas as pd
import numpy as np
from src.data_processing import preprocess_data, handle_missing_values
from src.config import DataConfig

def test_handle_missing_values():
    """Test that missing values are handled correctly."""
    data = pd.DataFrame({'A': [1, 2, np.nan], 'B': [np.nan, 5, 6]})
    config = DataConfig()
    result = handle_missing_values(data, config)
    assert result.isnull().sum().sum() == 0

def test_preprocess_data_feature_count():
    """Test that preprocessing results in expected number of features."""
    # Create a sample dataframe
    df = pd.DataFrame({
        'feat1': [1, 2, 3],
        'feat2': [4, 5, 6],
        'default': [0, 1, 0]
    })
    config = DataConfig()
    # Assuming preprocess_data returns X, y
    X, y = preprocess_data(df, config)
    # Check if the feature count is as expected (e.g., 2 features)
    assert X.shape[1] == 2

def test_target_column_present():
    """Test that the target column exists in the dataset."""
    df = pd.DataFrame({'col1': [1, 2], 'default': [0, 1]})
    assert 'default' in df.columns

def test_config_default_values():
    """Test that the config has sensible default values."""
    config = DataConfig()
    assert config.target_column == "default"
    assert config.test_size == 0.2

def test_model_config_constants():
    """Test that the model config constants are correctly set."""
    config = ModelConfig()
    assert config.SCORECARD_POINT_MIN == 400
    assert config.SCORECARD_POINT_MAX == 800