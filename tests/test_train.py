import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.train import (
    train_model,
    evaluate_model,
    run_training_pipeline,
    ModelConfig
)


def test_train_model_logistic_regression():
    """Test training logistic regression model."""
    X_train = pd.DataFrame({
        'feature1': np.random.randn(100),
        'feature2': np.random.randn(100)
    })
    y_train = pd.Series(np.random.randint(0, 2, size=100))
    
    config = ModelConfig(model_type="logistic_regression")
    model = train_model(X_train, y_train, config)
    
    assert model is not None
    assert hasattr(model, 'predict')
    assert hasattr(model, 'predict_proba')


def test_train_model_gradient_boosting():
    """Test training gradient boosting model."""
    X_train = pd.DataFrame({
        'feature1': np.random.randn(100),
        'feature2': np.random.randn(100)
    })
    y_train = pd.Series(np.random.randint(0, 2, size=100))
    
    config = ModelConfig(model_type="gradient_boosting")
    model = train_model(X_train, y_train, config)
    
    assert model is not None
    assert hasattr(model, 'predict')
    assert hasattr(model, 'predict_proba')


def test_evaluate_model_returns_metrics():
    """Test that evaluate_model returns expected metrics dictionary."""
    # Create mock model
    class MockModel:
        def predict(self, X):
            return np.array([0, 1, 0, 1, 0])
        
        def predict_proba(self, X):
            proba = np.random.rand(len(X), 2)
            proba = proba / proba.sum(axis=1, keepdims=True)
            return proba
    
    X_test = pd.DataFrame({'feature': range(5)})
    y_test = pd.Series([0, 1, 0, 1, 0])
    
    model = MockModel()
    metrics = evaluate_model(model, X_test, y_test)
    
    assert 'accuracy' in metrics
    assert 'roc_auc' in metrics
    assert 'gini' in metrics
    assert 'true_positives' in metrics
    assert 'false_positives' in metrics
    assert 'true_negatives' in metrics
    assert 'false_negatives' in metrics


def test_run_training_pipeline_returns_results():
    """Test that training pipeline runs and returns expected structure."""
    # Create minimal test data
    test_data = pd.DataFrame({
        'feature1': np.random.randn(50),
        'feature2': np.random.randn(50),
        'default': np.random.randint(0, 2, size=50)
    })
    test_path = "tests/test_dataset.csv"
    test_data.to_csv(test_path, index=False)
    
    try:
        config = ModelConfig(
            data_path=test_path,
            test_size=0.3,
            model_type="logistic_regression",
            experiment_name="test_experiment"
        )
        
        result = run_training_pipeline(config, log_mlflow=False)
        
        assert 'model' in result
        assert 'metrics' in result
        assert 'splits' in result
        assert 'config' in result
        assert 'roc_auc' in result['metrics']
        
    finally:
        Path(test_path).unlink()