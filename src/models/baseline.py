import pathlib
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

if __package__ is None:
    sys.path.insert(0, str(pathlib.Path(__file__).parents[2]))

from src.data.loader import DATA_DIR
from src.features.engineering import create_features, select_features

_FEATURES_PATH = DATA_DIR / "features.csv"
_CLEANED_PATH = DATA_DIR / "cleaned.csv"
_MODEL_PATH = DATA_DIR.parent / "models" / "baseline.pkl"
_TARGET_HINTS = {"class", "target", "label", "y"}


def _detect_target(df: pd.DataFrame) -> str:
    match = next((c for c in df.columns if c.lower() in _TARGET_HINTS), None)
    if match is None:
        raise ValueError(f"No target column found. Expected one of: {_TARGET_HINTS}")
    return match


def _is_classification(series: pd.Series) -> bool:
    return pd.api.types.is_integer_dtype(series) and series.nunique() < 20


def load_features(path: pathlib.Path = _FEATURES_PATH) -> pd.DataFrame:
    if path.exists():
        print(f"Loading features from {path.name}")
        return pd.read_csv(path)

    print(f"{path.name} not found — building from cleaned.csv ...")
    cleaned = pd.read_csv(_CLEANED_PATH)
    engineered = create_features(cleaned)
    _, df = select_features(engineered)
    df.to_csv(path, index=False)
    print(f"Saved {path}\n")
    return df


def train_and_evaluate(df: pd.DataFrame) -> dict:
    target_col = _detect_target(df)
    feature_cols = [c for c in df.columns if c != target_col]
    X, y = df[feature_cols], df[target_col]
    classification = _is_classification(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y if classification else None,
    )

    if classification:
        # StandardScaler is required for lbfgs to converge: features span
        # very different ranges (Time: 0–172k vs V-features: ±5).
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, random_state=42)),
        ])
        task = "classification"
    else:
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("reg", LinearRegression()),
        ])
        task = "regression"

    print(f"Task     : {task}")
    print(f"Target   : {target_col}")
    print(f"Features : {len(feature_cols)}")
    print(f"Train    : {len(X_train):,}  |  Test : {len(X_test):,}\n")

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    if classification:
        y_prob = model.predict_proba(X_test)[:, 1]
        metrics = {
            "accuracy" : accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall"   : recall_score(y_test, y_pred, zero_division=0),
            "f1"       : f1_score(y_test, y_pred, zero_division=0),
            "roc_auc"  : roc_auc_score(y_test, y_prob),
        }
        print("Classification metrics (test set):")
        for name, val in metrics.items():
            print(f"  {name:<12} {val:.4f}")
    else:
        metrics = {
            "mae" : mean_absolute_error(y_test, y_pred),
            "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
            "r2"  : r2_score(y_test, y_pred),
        }
        print("Regression metrics (test set):")
        for name, val in metrics.items():
            print(f"  {name:<12} {val:.4f}")

    _MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, _MODEL_PATH)
    print(f"\nModel saved → {_MODEL_PATH}")

    return {"model": model, "metrics": metrics, "task": task}


if __name__ == "__main__":
    df = load_features()
    print(f"Loaded : {df.shape[0]:,} rows, {df.shape[1]} columns\n")
    train_and_evaluate(df)
