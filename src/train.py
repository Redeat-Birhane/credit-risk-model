import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from src.data_processing import (
    build_feature_pipeline,
    apply_woe,
    build_numeric_pipeline,
    assign_high_risk_label
)

# =========================
# LOAD DATA
# =========================
df = pd.read_csv("data/raw/data.csv")

# =========================
# CREATE TARGET (TASK 4 OUTPUT)
# =========================
df = assign_high_risk_label(df)[0]  # labeled dataset

X = df.drop(columns=["is_high_risk"])
y = df["is_high_risk"]

# =========================
# FEATURE ENGINEERING
# =========================
feature_pipe = build_feature_pipeline()
X = feature_pipe.fit_transform(X)

# =========================
# SPLIT DATA
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# =========================
# MODEL LIST
# =========================
models = {
    "log_reg": LogisticRegression(max_iter=500),
    "decision_tree": DecisionTreeClassifier(random_state=42),
    "random_forest": RandomForestClassifier(n_estimators=100, random_state=42)
}

# =========================
# MLflow SETUP
# =========================
mlflow.set_experiment("credit-risk-models")

def evaluate_model(model, X_test, y_test):
    preds = model.predict(X_test)

    return {
        "accuracy": accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds),
        "recall": recall_score(y_test, preds),
        "f1": f1_score(y_test, preds),
        "roc_auc": roc_auc_score(y_test, preds)
    }

best_model = None
best_score = 0

# =========================
# TRAIN LOOP
# =========================
for name, model in models.items():

    with mlflow.start_run(run_name=name):

        model.fit(X_train, y_train)
        metrics = evaluate_model(model, X_test, y_test)

        # log params
        mlflow.log_param("model_name", name)

        # log metrics
        mlflow.log_metrics(metrics)

        # log model
        mlflow.sklearn.log_model(model, "model")

        print(f"{name} metrics:", metrics)

        # track best model (F1 used)
        if metrics["f1"] > best_score:
            best_score = metrics["f1"]
            best_model = model

# =========================
# REGISTER BEST MODEL
# =========================
mlflow.sklearn.log_model(
    best_model,
    artifact_path="best_model",
    registered_model_name="CreditRiskModel"
)

print("Best model registered successfully!")