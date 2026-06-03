from fastapi import FastAPI
import mlflow
import pandas as pd

from src.api.pydantic_models import PredictionRequest, PredictionResponse

app = FastAPI(title="Credit Risk API")

# =========================
# LOAD MODEL FROM MLFLOW
# =========================
MODEL_NAME = "CreditRiskModel"
MODEL_STAGE = "Production"

model = mlflow.sklearn.load_model("runs:/<RUN_ID>/model")

@app.get("/")
def health():
    return {"status": "API running"}


# =========================
# PREDICT ENDPOINT
# =========================
@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):

    # convert input to dataframe
    df = pd.DataFrame([request.data])

    # prediction probability
    prob = model.predict(df)[0]

    return PredictionResponse(
        risk_probability=float(prob),
        is_high_risk=int(prob > 0.5)
    )