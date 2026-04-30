"""
Session-scoped fixtures that generate synthetic data and model artifacts
when the real files are absent (i.e. in CI where data/ and models/ are
gitignored). Local runs with real artifacts are unaffected — the fixtures
only write files that don't already exist.
"""
import pathlib
import sys

import joblib
import numpy as np
import pandas as pd
import pytest
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

ROOT = pathlib.Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))

V_COLS = [f"V{i}" for i in range(1, 29)]

# Feature columns that match what create_features() adds
ENGINEERED_COLS = [
    "amount_log", "amount_rounded", "amount_small", "amount_large",
    "time_hour", "is_nighttime",
    "v_mean", "v_std", "v_abs_max", "top_fraud_signal",
    "v14_x_v12", "amount_log_x_v14", "v17_x_v12",
]
ALL_FEATURE_COLS = ["Time", "Amount"] + V_COLS + ENGINEERED_COLS


def _make_synthetic_cleaned(n_legit=4975, n_fraud=25, seed=42) -> pd.DataFrame:
    """
    Synthetic credit-card dataset with a strong V14 fraud signal.
    V14 for fraud ~ N(-6, 0.5); for legit ~ N(0, 1).
    Fraud rate is 0.5% — enough to trigger the imbalance warning in tests.
    """
    rng = np.random.default_rng(seed)
    n = n_legit + n_fraud

    data = {}
    for col in V_COLS:
        legit_vals = rng.standard_normal(n_legit)
        fraud_vals = rng.standard_normal(n_fraud)
        if col == "V14":
            fraud_vals = fraud_vals * 0.5 - 6.0   # strong fraud signal
        elif col in ("V17", "V12"):
            fraud_vals = fraud_vals * 0.7 - 3.0   # secondary signals
        data[col] = np.concatenate([legit_vals, fraud_vals])

    data["Time"]   = rng.uniform(0, 172_800, n)
    data["Amount"] = np.concatenate([
        rng.uniform(1, 500, n_legit),
        rng.uniform(1, 10,  n_fraud),   # small amounts for fraud
    ])
    data["Class"] = np.array([0] * n_legit + [1] * n_fraud)

    df = pd.DataFrame(data)
    return df.sample(frac=1, random_state=seed).reset_index(drop=True)


def _make_synthetic_features(cleaned: pd.DataFrame) -> pd.DataFrame:
    """Run the real create_features() pipeline on the synthetic data."""
    from src.features.engineering import create_features
    return create_features(cleaned)


def _train_synthetic_model(features: pd.DataFrame) -> ImbPipeline:
    """Train a fast LogisticRegression pipeline on synthetic features."""
    X = features[[c for c in ALL_FEATURE_COLS if c in features.columns]]
    y = features["Class"]

    pipeline = ImbPipeline([
        ("scaler", StandardScaler()),
        ("smote",  SMOTE(sampling_strategy=0.1, random_state=42, k_neighbors=3)),
        ("clf",    LogisticRegression(max_iter=1000, random_state=42, C=0.1)),
    ])
    pipeline.fit(X, y)
    return pipeline


@pytest.fixture(scope="session", autouse=True)
def ensure_artifacts():
    """
    Generate synthetic data and model files if the real ones are absent.
    Runs automatically at the start of every test session.
    """
    cleaned_path  = ROOT / "data" / "cleaned.csv"
    features_path = ROOT / "data" / "features.csv"
    tuned_path    = ROOT / "models" / "tuned_model.pkl"
    baseline_path = ROOT / "models" / "baseline.pkl"

    need_data     = not cleaned_path.exists() or not features_path.exists()
    need_tuned    = not tuned_path.exists()
    need_baseline = not baseline_path.exists()

    # If data is missing, synthetic models must be regenerated too — mixing
    # real data with synthetic models (or vice versa) produces inconsistent state.
    if need_data:
        need_tuned    = True
        need_baseline = True

    if not need_data and not need_tuned and not need_baseline:
        return   # all artifacts present — nothing to do

    print("\n[conftest] Generating synthetic fixtures for CI...")

    (ROOT / "data").mkdir(exist_ok=True)
    (ROOT / "models").mkdir(exist_ok=True)

    if need_data:
        cleaned = _make_synthetic_cleaned()
        cleaned.to_csv(cleaned_path, index=False)
        print(f"[conftest]   wrote {cleaned_path.name}  ({len(cleaned):,} rows)")

        features = _make_synthetic_features(cleaned)
        features.to_csv(features_path, index=False)
        print(f"[conftest]   wrote {features_path.name}  ({features.shape[1]} cols)")
    else:
        features = pd.read_csv(features_path)

    if need_tuned or need_baseline:
        model = _train_synthetic_model(features)
        if need_tuned:
            joblib.dump(model, tuned_path)
            print("[conftest]   wrote tuned_model.pkl")
        if need_baseline:
            joblib.dump(model, baseline_path)
            print("[conftest]   wrote baseline.pkl")
