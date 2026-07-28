from dataclasses import dataclass
from typing import Optional

@dataclass
class ModelConfig:
    """Configuration for the machine learning model."""
    model_type: str = "logistic_regression"
    random_state: int = 42
    test_size: float = 0.2
    # Named constants replace magic numbers
    SCORECARD_POINT_MIN: int = 400
    SCORECARD_POINT_MAX: int = 800
    SCORECARD_ODDS_DOUBLE: int = 20
    SCORECARD_PDO: int = 20

@dataclass
class DataConfig:
    """Configuration for data paths and processing."""
    raw_data_path: str = "data/raw/"
    processed_data_path: str = "data/processed/"
    target_column: str = "default"
    # Named constants for feature groups
    RFMS_CATEGORIES: tuple = ("Debit", "Consumption", "Transfer", "Phone Bill", "Utility Bill", "Gaming")
    MONTHS_TO_AGGREGATE: int = 6

@dataclass
class APIConfig:
    """Configuration for the FastAPI service."""
    host: str = "0.0.0.0"
    port: int = 8000
    model_path: str = "models/trained_model.pkl"