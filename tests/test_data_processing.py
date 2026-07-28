import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.data_processing import (
    load_data,
    clean_data,
    engineer_features,
    split_data,
    calculate_woe_iv,
    ModelConfig
)


def test_load_data_valid():
    """Test loading valid data file."""
    # Create a minimal test CSV
    test_data = pd.DataFrame({
        'feature1': [1, 2, 3],
        'feature2': [4, 5, 6],
        'default': [0, 1, 0]
    })
    test_path = "tests/test_data.csv"
    test_data.to_csv(test_path, index=False)
    
    # Test load
    df = load_data(test_path)
    assert len(df) == 3
    assert 'feature1' in df.columns
    
    # Cleanup
    Path(test_path).unlink()


def test_clean_data_removes_duplicates():
    """Test that clean_data removes duplicate rows."""
    df = pd.DataFrame({
        'feature': [1, 2, 2, 3],
        'default': [0, 1, 1, 0]
    })
    cleaned = clean_data(df)
    assert len(cleaned) == 3
    assert cleaned['feature'].tolist() == [1, 2, 3]


def test_clean_data_handles_missing_values():
    """Test that clean_data handles missing values appropriately."""
    df = pd.DataFrame({
        'feature1': [1, 2, np.nan, 4],
        'feature2': [5, np.nan, 7, 8],
        'default': [0, 1, 0, 1]
    })
    cleaned = clean_data(df)
    assert not cleaned.isnull().any().any()


def test_engineer_features_raises_error_without_target():
    """Test that engineer_features raises ValueError when target is missing."""
    df = pd.DataFrame({
        'feature1': [1, 2, 3],
        'feature2': [4, 5, 6]
    })
    with pytest.raises(ValueError, match="Target column 'default' not found"):
        engineer_features(df)


def test_engineer_features_returns_correct_shapes():
    """Test that engineer_features returns features and target with correct shapes."""
    df = pd.DataFrame({
        'feature1': [1, 2, 3, 4],
        'feature2': [5, 6, 7, 8],
        'default': [0, 1, 0, 1]
    })
    X, y = engineer_features(df)
    assert X.shape[0] == 4
    assert y.shape[0] == 4
    assert 'default' not in X.columns


def test_split_data_creates_correct_splits():
    """Test that split_data produces train/test splits correctly."""
    X = pd.DataFrame({'feature': range(100)})
    y = pd.Series([0, 1] * 50)
    
    splits = split_data(X, y, test_size=0.2, random_state=42)
    
    assert 'X_train' in splits
    assert 'X_test' in splits
    assert len(splits['X_train']) == 80
    assert len(splits['X_test']) == 20


def test_calculate_woe_iv_valid_input():
    """Test that calculate_woe_iv handles valid input correctly."""
    df = pd.DataFrame({
        'feature': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'default': [0, 0, 1, 0, 1, 0, 1, 1, 0, 1]
    })
    
    result_df, woe_values, iv = calculate_woe_iv(df, 'feature', 'default', bins=3)
    
    assert len(result_df) > 0
    assert isinstance(iv, float)
    assert len(woe_values) > 0


def test_calculate_woe_iv_invalid_target():
    """Test that calculate_woe_iv raises error for invalid target values."""
    df = pd.DataFrame({
        'feature': [1, 2, 3],
        'default': [0, 2, 1]  # Invalid value '2'
    })
    with pytest.raises(ValueError, match="Target must be binary"):
        calculate_woe_iv(df, 'feature', 'default')


def test_model_config_dataclass_defaults():
    """Test that ModelConfig creates instance with default values."""
    config = ModelConfig()
    assert config.data_path == "data/raw/credit_data.csv"
    assert config.test_size == 0.2
    assert config.random_state == 42
    assert config.model_type == "logistic_regression"
    assert config.experiment_name == "credit_risk_model"


def test_model_config_custom_values():
    """Test that ModelConfig accepts custom values."""
    config = ModelConfig(
        data_path="custom/data.csv",
        test_size=0.3,
        model_type="gradient_boosting"
    )
    assert config.data_path == "custom/data.csv"
    assert config.test_size == 0.3
    assert config.model_type == "gradient_boosting"