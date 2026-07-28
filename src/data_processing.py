"""
Data processing module for credit risk modeling.
Handles loading, cleaning, and feature engineering.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional, Dict, List, Union
from pathlib import Path
from dataclasses import dataclass
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split


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
    
    # Drop columns that are not useful for modeling
    # These are identifiers that don't add predictive value
    columns_to_drop = ['TransactionId', 'BatchId', 'AccountId', 'SubscriptionId', 
                      'CustomerId', 'TransactionStartTime']
    df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])
    
    return df


def engineer_features(
    df: pd.DataFrame,
    target_column: str = 'FraudResult',  # Changed from 'default' to 'FraudResult'
    categorical_columns: Optional[List[str]] = None,
    scale_numeric: bool = True
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Engineer features for credit risk/fraud modeling.
    
    Args:
        df: Cleaned dataframe
        target_column: Name of the target column (default: 'FraudResult')
        categorical_columns: List of categorical column names to encode
        scale_numeric: Whether to standardize numeric features
        
    Returns:
        Tuple[pd.DataFrame, pd.Series]: Feature matrix and target variable
        
    Raises:
        ValueError: If target column is not found
    """
    # Check if target exists
    if target_column not in df.columns:
        # If not found, try to find it
        possible_targets = ['FraudResult', 'default', 'target', 'y', 'label', 'status']
        found_target = None
        for col in possible_targets:
            if col in df.columns:
                found_target = col
                break
        
        if found_target is None:
            raise ValueError(f"Target column not found. Available columns: {list(df.columns)}")
        else:
            print(f"   ✅ Using '{found_target}' as target column")
            target_column = found_target
    
    # Separate features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    # Identify categorical columns (object type or low cardinality)
    if categorical_columns is None:
        categorical_columns = X.select_dtypes(include=['object']).columns.tolist()
        # Also include integer columns with low cardinality
        int_cols = X.select_dtypes(include=['int64']).columns
        for col in int_cols:
            if X[col].nunique() < 10:  # Low cardinality integer columns
                categorical_columns.append(col)
    
    # Encode categorical variables
    if categorical_columns:
        print(f"   📝 Encoding categorical columns: {categorical_columns}")
        X = pd.get_dummies(X, columns=categorical_columns, drop_first=True)
    
    # Scale numeric features (optional)
    if scale_numeric:
        scaler = StandardScaler()
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            X[numeric_cols] = scaler.fit_transform(X[numeric_cols])
            print(f"   🔢 Scaled {len(numeric_cols)} numeric features")
    
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
    target: str = 'FraudResult',  # Updated to 'FraudResult'
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


@dataclass
class ModelConfig:
    """Configuration for credit risk model."""
    
    # Data parameters
    data_path: str = "data/raw/data.csv"
    processed_data_path: str = "data/processed/credit_data_processed.csv"
    target_column: str = "FraudResult"  # Updated to 'FraudResult'
    test_size: float = 0.2
    random_state: int = 42
    
    # Model parameters
    model_type: str = "logistic_regression"
    max_iter: int = 1000
    regularization: str = "l2"
    C: float = 1.0
    
    # Training parameters
    cv_folds: int = 5
    scoring_metric: str = "roc_auc"
    
    # MLflow tracking
    experiment_name: str = "credit_risk_model"
    run_name: Optional[str] = None


def main():
    """Main entry point for data processing pipeline."""
    print("="*60)
    print("🚀 Starting Data Processing Pipeline")
    print("="*60)
    
    # Load configuration
    config = ModelConfig()
    
    # 1. Load raw data
    print(f"\n📂 Loading raw data from: {config.data_path}")
    try:
        df = load_data(config.data_path)
        print(f"   ✅ Loaded {len(df)} rows, {len(df.columns)} columns")
        print(f"   📋 Columns: {list(df.columns)}")
    except FileNotFoundError:
        print(f"   ❌ File not found. Looking for data in other locations...")
        alt_paths = ["data/raw/data.csv", "data/data.csv", "data.csv"]
        found = False
        for path in alt_paths:
            try:
                df = load_data(path)
                print(f"   ✅ Found data at: {path}")
                found = True
                break
            except FileNotFoundError:
                continue
        if not found:
            print("   ❌ No data file found. Please ensure your data is in data/raw/data.csv")
            return
    
    # 2. Clean data
    print("\n🧹 Cleaning data...")
    df = clean_data(df)
    print(f"   ✅ Cleaned data: {len(df)} rows, {len(df.columns)} columns")
    
    # 3. Engineer features
    print("\n🔧 Engineering features...")
    try:
        X, y = engineer_features(
            df, 
            target_column=config.target_column,
            categorical_columns=None, 
            scale_numeric=True
        )
        print(f"   ✅ Features engineered: {X.shape[1]} features")
        print(f"   ✅ Target shape: {y.shape}")
        print(f"   ✅ Target distribution: 0={sum(y==0)}, 1={sum(y==1)} ({sum(y==1)/len(y):.2%})")
    except ValueError as e:
        print(f"   ❌ Error: {e}")
        return
    
    # 4. Split data
    print("\n📊 Splitting data into train/test...")
    split_data_result = split_data(X, y, test_size=config.test_size, random_state=config.random_state)
    print(f"   ✅ Train size: {len(split_data_result['X_train'])}")
    print(f"   ✅ Test size: {len(split_data_result['X_test'])}")
    
    # 5. Save processed data
    print(f"\n💾 Saving processed data to: {config.processed_data_path}")
    import os
    os.makedirs('data/processed', exist_ok=True)
    
    # Combine X and y back together for saving
    processed_df = X.copy()
    processed_df[config.target_column] = y
    processed_df.to_csv(config.processed_data_path, index=False)
    print(f"   ✅ Saved processed data!")
    
    # 6. Calculate some statistics
    print("\n📊 Data Summary:")
    print(f"   - Total samples: {len(processed_df)}")
    print(f"   - Total features: {len(processed_df.columns)}")
    print(f"   - Fraud rate: {y.mean():.2%}")
    print(f"   - Features: {list(X.columns[:5])}{'...' if len(X.columns) > 5 else ''}")
    
    print("\n" + "="*60)
    print("✅ Data Processing Complete!")
    print("="*60)


if __name__ == "__main__":
    main()