"""
Model explainability module using SHAP.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from typing import Tuple, Any, Optional, Dict, List
import pickle
from pathlib import Path


def explain_with_shap(
    model: Any,
    X_train: pd.DataFrame,
    X_test: Optional[pd.DataFrame] = None,
    model_type: str = "tree"
) -> Tuple[Any, Any, Optional[Any]]:
    """
    Generate SHAP explanations for model predictions.
    
    Args:
        model: Trained model
        X_train: Training data for explainer creation
        X_test: Test data for explanations (optional)
        model_type: Type of model ('tree' for XGBoost, 'linear' for sklearn)
        
    Returns:
        Tuple of (explainer, shap_values_test, shap_values_train)
    """
    if model_type == "tree":
        explainer = shap.TreeExplainer(model)
    elif model_type == "linear":
        explainer = shap.LinearExplainer(model, X_train)
    else:
        explainer = shap.KernelExplainer(model.predict_proba, X_train)
    
    shap_values_train = explainer.shap_values(X_train)
    
    shap_values_test = None
    if X_test is not None:
        shap_values_test = explainer.shap_values(X_test)
    
    return explainer, shap_values_test, shap_values_train


def plot_shap_summary(
    shap_values: np.ndarray,
    X_data: pd.DataFrame,
    title: str = "SHAP Feature Importance"
) -> plt.Figure:
    """
    Create SHAP summary plot.
    
    Args:
        shap_values: SHAP values from explainer
        X_data: Feature data
        title: Plot title
        
    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    shap.summary_plot(
        shap_values,
        X_data,
        show=False,
        plot_type="dot"
    )
    
    plt.title(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    return fig


def plot_shap_bar(
    shap_values: np.ndarray,
    X_data: pd.DataFrame,
    title: str = "Mean SHAP Values"
) -> plt.Figure:
    """
    Create SHAP bar plot for global importance.
    
    Args:
        shap_values: SHAP values from explainer
        X_data: Feature data
        title: Plot title
        
    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    shap.summary_plot(
        shap_values,
        X_data,
        show=False,
        plot_type="bar"
    )
    
    plt.title(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    return fig


def plot_shap_waterfall(
    explainer: Any,
    shap_values: np.ndarray,
    X_row: pd.Series,
    feature_names: Optional[List[str]] = None
) -> plt.Figure:
    """
    Create SHAP waterfall plot for a single prediction.
    
    Args:
        explainer: SHAP explainer
        shap_values: SHAP values for the specific row
        X_row: Single row of feature data
        feature_names: Optional list of feature names
        
    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    shap.waterfall_plot(
        shap.Explanation(
            values=shap_values,
            base_values=explainer.expected_value,
            data=X_row.values,
            feature_names=feature_names or X_row.index.tolist()
        ),
        show=False
    )
    
    plt.tight_layout()
    return fig


def plot_shap_dependence(
    shap_values: np.ndarray,
    X_data: pd.DataFrame,
    feature_name: str,
    color_feature: Optional[str] = None
) -> plt.Figure:
    """
    Create SHAP dependence plot for a specific feature.
    
    Args:
        shap_values: SHAP values
        X_data: Feature data
        feature_name: Feature to plot
        color_feature: Feature to color points by (optional)
        
    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    shap.dependence_plot(
        feature_name,
        shap_values,
        X_data,
        interaction_index=color_feature,
        show=False
    )
    
    plt.title(f"SHAP Dependence Plot: {feature_name}", fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    return fig


def generate_shap_report(
    model: Any,
    X_train: pd.DataFrame,
    X_test: Optional[pd.DataFrame] = None,
    output_dir: str = "reports/figures/",
    model_type: str = "tree"
) -> Dict[str, Any]:
    """
    Generate comprehensive SHAP explanation report.
    
    Args:
        model: Trained model
        X_train: Training features
        X_test: Test features (optional)
        output_dir: Directory to save plots
        model_type: Type of model
        
    Returns:
        Dictionary with SHAP values and explainer
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Generate explanations
    explainer, shap_values_test, shap_values_train = explain_with_shap(
        model, X_train, X_test, model_type
    )
    
    results = {
        'explainer': explainer,
        'shap_values_train': shap_values_train,
        'shap_values_test': shap_values_test
    }
    
    # Generate and save plots
    if shap_values_test is not None:
        # Summary plot
        fig_summary = plot_shap_summary(
            shap_values_test,
            X_test,
            "SHAP Summary - Credit Risk Model"
        )
        fig_summary.savefig(f"{output_dir}/shap_summary.png", dpi=300, bbox_inches='tight')
        plt.close(fig_summary)
        
        # Bar plot
        fig_bar = plot_shap_bar(
            shap_values_test,
            X_test,
            "Global Feature Importance"
        )
        fig_bar.savefig(f"{output_dir}/shap_bar.png", dpi=300, bbox_inches='tight')
        plt.close(fig_bar)
        
        # Dependence plots for top features
        top_features = np.abs(shap_values_test).mean(0).argsort()[-5:][::-1]
        for idx in top_features:
            feature_name = X_test.columns[idx]
            fig_dep = plot_shap_dependence(
                shap_values_test,
                X_test,
                feature_name
            )
            fig_dep.savefig(
                f"{output_dir}/shap_dependence_{feature_name}.png",
                dpi=300,
                bbox_inches='tight'
            )
            plt.close(fig_dep)
    
    # Save SHAP values for later use
    with open(f"{output_dir}/shap_values.pkl", 'wb') as f:
        pickle.dump(results, f)
    
    return results


# Example usage
if __name__ == "__main__":
    # Load data and model
    from src.train import load_model
    
    # This is a demonstration - adjust paths as needed
    model = load_model("models/model.pkl")
    
    # Load sample data
    X_sample = pd.DataFrame(np.random.randn(100, 10), 
                           columns=[f'feature_{i}' for i in range(10)])
    
    # Generate explanations
    results = generate_shap_report(
        model,
        X_sample,
        X_sample.head(50),
        model_type="linear"
    )
    
    print("✅ SHAP explanations saved to reports/figures/")