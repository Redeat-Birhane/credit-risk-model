"""
Model training module for credit risk prediction.
"""

import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    roc_auc_score, 
    accuracy_score, 
    classification_report,
    confusion_matrix
)
from sklearn.model_selection import cross_val_score
import joblib
from pathlib import Path
from src.data_processing import ModelConfig, split_data, engineer_features


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    config: ModelConfig,
    model_type: Optional[str] = None
) -> Any:
    """
    Train credit risk model based on configuration.
    
    Args:
        X_train: Training features
        y_train: Training target
        config: Model configuration object
        model_type: Override model type (default: from config)
        
    Returns:
        Trained model instance
    """
    model_type = model_type or config.model_type
    
    if model_type == "logistic_regression":
        model = LogisticRegression(
            max_iter=config.max_iter,
            penalty=config.regularization,
            C=config.C,
            random_state=config.random_state,
            class_weight='balanced'
        )
    elif model_type == "gradient_boosting":
        model = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            random_state=config.random_state,
            subsample=0.8
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    model.fit(X_train, y_train)
    return model


def evaluate_model(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> Dict[str, float]:
    """
    Evaluate model performance using multiple metrics.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test target
        
    Returns:
        Dictionary of evaluation metrics
    """
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_pred_proba),
        'gini': 2 * roc_auc_score(y_test, y_pred_proba) - 1,
    }
    
    # Add confusion matrix components
    cm = confusion_matrix(y_test, y_pred)
    metrics['true_positives'] = int(cm[1, 1])
    metrics['false_positives'] = int(cm[0, 1])
    metrics['true_negatives'] = int(cm[0, 0])
    metrics['false_negatives'] = int(cm[1, 0])
    
    return metrics


def run_training_pipeline(
    config: ModelConfig,
    log_mlflow: bool = True
) -> Dict[str, Any]:
    """
    Execute complete training pipeline with MLflow tracking.
    
    Args:
        config: Model configuration
        log_mlflow: Whether to log to MLflow
        
    Returns:
        Dictionary containing model, metrics, and data splits
    """
    # Load and preprocess data
    from src.data_processing import load_data, clean_data
    
    df = load_data(config.data_path)
    df = clean_data(df)
    
    # Engineer features
    X, y = engineer_features(
        df,
        categorical_columns=['registration_channel', 'bank_type']
    )
    
    # Split data
    splits = split_data(X, y, config.test_size, config.random_state)
    
    # Train model
    model = train_model(
        splits['X_train'],
        splits['y_train'],
        config
    )
    
    # Evaluate model
    metrics = evaluate_model(model, splits['X_test'], splits['y_test'])
    
    # Cross-validation
    cv_scores = cross_val_score(
        model,
        splits['X_train'],
        splits['y_train'],
        cv=config.cv_folds,
        scoring=config.scoring_metric
    )
    metrics['cv_mean'] = cv_scores.mean()
    metrics['cv_std'] = cv_scores.std()
    
    # Log to MLflow
    if log_mlflow:
        mlflow.set_experiment(config.experiment_name)
        
        with mlflow.start_run(run_name=config.run_name):
            # Log parameters
            mlflow.log_params({
                'model_type': config.model_type,
                'test_size': config.test_size,
                'max_iter': config.max_iter,
                'regularization': config.regularization,
                'C': config.C
            })
            
            # Log metrics
            mlflow.log_metrics(metrics)
            
            # Log model
            mlflow.sklearn.log_model(model, "model")
            
            # Log artifacts
            mlflow.log_artifact(Path(config.data_path).parent / "processed")
    
    return {
        'model': model,
        'metrics': metrics,
        'splits': splits,
        'config': config
    }


def save_model(model: Any, path: str = "models/model.pkl") -> None:
    """Save trained model to disk."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load_model(path: str = "models/model.pkl") -> Any:
    """Load trained model from disk."""
    return joblib.load(path)


if __name__ == "__main__":
    # Example usage
    config = ModelConfig(
        data_path="data/processed/credit_data_processed.csv",
        model_type="logistic_regression"
    )
    
    result = run_training_pipeline(config)
    
    print("Model Training Complete!")
    print(f"ROC AUC: {result['metrics']['roc_auc']:.4f}")
    print(f"Gini Coefficient: {result['metrics']['gini']:.4f}")