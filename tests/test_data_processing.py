import pandas as pd
from sklearn.model_selection import train_test_split

from src.data_processing import (
    build_feature_pipeline,
    apply_woe,
    build_numeric_pipeline,
    get_iv_table,
    assign_high_risk_label
)

# =========================
# LOAD DATA
# =========================
df = pd.read_csv("data/raw/data.csv")

X = df.drop("FraudResult", axis=1)
y = df["FraudResult"]

# =========================
# STEP 1: FEATURE ENGINEERING
# =========================
feature_pipe = build_feature_pipeline()
X_features = feature_pipe.fit_transform(X)

# =========================
# STEP 2: SPLIT
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X_features, y, test_size=0.2, random_state=42
)

# =========================
# STEP 3: WOE ENCODING
# =========================
X_train_woe, X_test_woe, woe_model = apply_woe(
    X_train, X_test, y_train
)

# =========================
# STEP 4: SCALING
# =========================
num_pipe = build_numeric_pipeline()

X_train_final = num_pipe.fit_transform(X_train_woe)
X_test_final = num_pipe.transform(X_test_woe)

# =========================
# STEP 5: IV TABLE
# =========================
iv = get_iv_table(X_train_woe, y_train)

print(iv.sort_values("IV", ascending=False))

df_labeled, rfm, model = assign_high_risk_label(df)

print(df_labeled[["CustomerId", "is_high_risk"]].head())
print(df_labeled["is_high_risk"].value_counts())

def test_feature_pipeline_output():
    df = pd.read_csv("data/raw/data.csv").head(50)

    X = df.drop(columns=["FraudResult"], errors="ignore")

    pipe = build_feature_pipeline()
    X_out = pipe.fit_transform(X)

    assert X_out is not None
    assert X_out.shape[0] == X.shape[0]


def test_risk_label_creation():
    df = pd.read_csv("data/raw/data.csv").head(50)

    df_labeled, _, _ = assign_high_risk_label(df)

    assert "is_high_risk" in df_labeled.columns
    assert df_labeled["is_high_risk"].isin([0, 1]).all()