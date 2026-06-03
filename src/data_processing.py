# Aggregate Features Transformer
import pandas as pd
import numpy as np
from category_encoders import WOEEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

from xverse.transformer import WOE


# =========================
# FEATURE ENGINEERING
# =========================
class AggregateFeaturesTransformer(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        agg = X.groupby("CustomerId").agg(
            TotalTransactionAmount=("Amount", "sum"),
            AverageTransactionAmount=("Amount", "mean"),
            TransactionCount=("TransactionId", "count"),
            StdTransactionAmount=("Amount", "std"),
        ).reset_index()

        agg["StdTransactionAmount"] = agg["StdTransactionAmount"].fillna(0)

        return X.merge(agg, on="CustomerId", how="left")


# =========================
# DATETIME FEATURES
# =========================
class DateTimeFeaturesTransformer(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        X["TransactionStartTime"] = pd.to_datetime(X["TransactionStartTime"])

        X["TransactionHour"] = X["TransactionStartTime"].dt.hour
        X["TransactionDay"] = X["TransactionStartTime"].dt.day
        X["TransactionMonth"] = X["TransactionStartTime"].dt.month
        X["TransactionYear"] = X["TransactionStartTime"].dt.year

        return X


# =========================
# DROP COLUMNS
# =========================
class DropColumnsTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, columns=None):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.drop(columns=self.columns, errors="ignore")


# =========================
# COLUMN DEFINITIONS
# =========================
NUMERIC_FEATURES = [
    "CountryCode",
    "Amount",
    "Value",
    "PricingStrategy",
    "TotalTransactionAmount",
    "AverageTransactionAmount",
    "TransactionCount",
    "StdTransactionAmount",
    "TransactionHour",
    "TransactionDay",
    "TransactionMonth",
    "TransactionYear",
]

CATEGORICAL_FEATURES = [
    "ProviderId",
    "ProductId",
    "ProductCategory",
    "ChannelId",
    "CurrencyCode"
]

DROP_COLUMNS = [
    "TransactionId",
    "BatchId",
    "AccountId",
    "SubscriptionId",
    "CustomerId",
    "TransactionStartTime"
]


# =========================
# PIPELINE BUILDERS
# =========================
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])


preprocessor = ColumnTransformer([
    ("num", numeric_pipeline, NUMERIC_FEATURES),
    ("cat", categorical_pipeline, CATEGORICAL_FEATURES)
])


def build_feature_pipeline():
    return Pipeline(steps=[
        ("aggregate", AggregateFeaturesTransformer()),
        ("datetime", DateTimeFeaturesTransformer()),
        ("drop", DropColumnsTransformer(columns=DROP_COLUMNS)),
    ])




def apply_woe(X_train, X_test, y_train):
    X_train = X_train.copy()
    X_test = X_test.copy()

    # STEP 1: ensure all columns are usable by encoder
    for col in X_train.columns:
        if not pd.api.types.is_numeric_dtype(X_train[col]):
            X_train[col] = X_train[col].astype(str)

    for col in X_test.columns:
        if not pd.api.types.is_numeric_dtype(X_test[col]):
            X_test[col] = X_test[col].astype(str)

    # STEP 2: align columns
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

    # STEP 3: WOE encoding (stable implementation)
    woe = WOEEncoder(cols=X_train.columns)

    X_train_woe = woe.fit_transform(X_train, y_train)
    X_test_woe = woe.transform(X_test)

    return X_train_woe, X_test_woe, woe

    





# =========================
# IV TABLE
# =========================


def compute_iv(df, feature, target):
    eps = 0.0001

    temp = pd.DataFrame()
    temp["feature"] = df[feature]
    temp["target"] = target.values

    grouped = temp.groupby("feature")["target"].agg(["count", "sum"])
    grouped.columns = ["total", "bad"]

    grouped["good"] = grouped["total"] - grouped["bad"]

    grouped["bad_dist"] = grouped["bad"] / grouped["bad"].sum()
    grouped["good_dist"] = grouped["good"] / grouped["good"].sum()

    grouped["woe"] = np.log((grouped["good_dist"] + eps) / (grouped["bad_dist"] + eps))

    grouped["iv"] = (grouped["good_dist"] - grouped["bad_dist"]) * grouped["woe"]

    return grouped["iv"].sum()




def compute_iv_table(df, target):
    iv_list = []

    for col in df.columns:
        if col == target.name if hasattr(target, "name") else None:
            continue

        try:
            iv = compute_iv(df, col, target)
            iv_list.append((col, iv))
        except:
            continue

    return pd.DataFrame(iv_list, columns=["feature", "IV"]).sort_values(
        by="IV", ascending=False
    )
def get_iv_table(df, target):
    return compute_iv_table(df, target)


# =========================
# ACCESSORS
# =========================
def build_numeric_pipeline():
    return numeric_pipeline

def build_pipeline():
    return build_feature_pipeline()