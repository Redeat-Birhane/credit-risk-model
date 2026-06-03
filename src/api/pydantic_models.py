from pydantic import BaseModel
from typing import Dict, Any


class PredictionRequest(BaseModel):
    data: Dict[str, Any]


class PredictionResponse(BaseModel):
    risk_probability: float
    is_high_risk: int