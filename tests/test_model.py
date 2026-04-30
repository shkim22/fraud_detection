import sys
import pathlib

import numpy as np
import pandas as pd
import pytest

ROOT = pathlib.Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))

joblib     = pytest.importorskip("joblib")
xgboost    = pytest.importorskip("xgboost")

TUNED_PATH    = ROOT / "models" / "tuned_model.pkl"
BASELINE_PATH = ROOT / "models" / "baseline.pkl"
FEATURES_PATH = ROOT / "data" / "features.csv"
V_COLS = [f"V{i}" for i in range(1, 29)]

# Minimal feature vector matching the tuned model's expected columns
_FEATURE_DEFAULTS = {
    **{col: 0.0 for col in V_COLS},
    "Time": 0.0, "Amount": 150.0,
    "amount_log": np.log1p(150.0),
    "amount_rounded": 0, "amount_small": 0, "amount_large": 0,
    "time_hour": 14, "is_nighttime": 0,
    "v_mean": 0.0, "v_std": 0.0, "v_abs_max": 0.0,
    "top_fraud_signal": 0.0,
    "v14_x_v12": 0.0, "amount_log_x_v14": 0.0, "v17_x_v12": 0.0,
}


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def tuned_model():
    return joblib.load(TUNED_PATH)


@pytest.fixture(scope="module")
def baseline_model():
    return joblib.load(BASELINE_PATH)


@pytest.fixture(scope="module")
def feature_columns(tuned_model):
    """Derive the expected column order from the model's scaler step."""
    scaler = tuned_model.named_steps["scaler"]
    return list(scaler.feature_names_in_)


@pytest.fixture(scope="module")
def single_row(feature_columns):
    return pd.DataFrame([{c: _FEATURE_DEFAULTS.get(c, 0.0) for c in feature_columns}])


@pytest.fixture(scope="module")
def fraud_row(feature_columns):
    """Craft a row with strong fraud signals (V14 << 0)."""
    row = {c: _FEATURE_DEFAULTS.get(c, 0.0) for c in feature_columns}
    row.update({"V14": -8.0, "V17": -5.0, "V12": -3.0,
                "top_fraud_signal": (-8 - 5 - 3) / 3,
                "v14_x_v12": -8.0 * -3.0,
                "amount_log_x_v14": np.log1p(1.0) * -8.0,
                "v17_x_v12": -5.0 * -3.0,
                "Amount": 1.0, "amount_log": np.log1p(1.0),
                "amount_small": 1})
    return pd.DataFrame([row])


@pytest.fixture(scope="module")
def test_sample(feature_columns):
    """200-row stratified sample from features.csv."""
    df = pd.read_csv(FEATURES_PATH)
    legit  = df[df["Class"] == 0].sample(190, random_state=42)
    fraud  = df[df["Class"] == 1].sample(10,  random_state=42)
    sample = pd.concat([legit, fraud]).reset_index(drop=True)
    X = sample[[c for c in feature_columns if c in sample.columns]]
    y = sample["Class"]
    return X, y


# ── Model loads ───────────────────────────────────────────────────────────────

def test_tuned_model_loads():
    model = joblib.load(TUNED_PATH)
    assert model is not None


def test_baseline_model_loads():
    model = joblib.load(BASELINE_PATH)
    assert model is not None


def test_tuned_model_has_expected_steps(tuned_model):
    assert "scaler" in tuned_model.named_steps
    assert "clf"    in tuned_model.named_steps


# ── Single-row predictions ────────────────────────────────────────────────────

def test_predict_returns_array(tuned_model, single_row):
    pred = tuned_model.predict(single_row)
    assert len(pred) == 1


def test_predict_proba_returns_two_classes(tuned_model, single_row):
    proba = tuned_model.predict_proba(single_row)
    assert proba.shape == (1, 2)


def test_predict_proba_sums_to_one(tuned_model, single_row):
    proba = tuned_model.predict_proba(single_row)
    np.testing.assert_almost_equal(proba.sum(axis=1)[0], 1.0, decimal=6)


def test_fraud_probability_in_unit_interval(tuned_model, single_row):
    fraud_prob = tuned_model.predict_proba(single_row)[0, 1]
    assert 0.0 <= fraud_prob <= 1.0


def test_prediction_is_binary(tuned_model, single_row):
    pred = tuned_model.predict(single_row)[0]
    assert pred in (0, 1)


# ── Directional sanity ────────────────────────────────────────────────────────

def test_normal_transaction_has_low_fraud_prob(tuned_model, single_row):
    fraud_prob = tuned_model.predict_proba(single_row)[0, 1]
    assert fraud_prob < 0.3, f"Expected low fraud prob for normal tx, got {fraud_prob:.4f}"


def test_fraud_signals_raise_fraud_prob(tuned_model, single_row, fraud_row):
    normal_prob = tuned_model.predict_proba(single_row)[0, 1]
    fraud_prob  = tuned_model.predict_proba(fraud_row)[0, 1]
    assert fraud_prob > normal_prob, (
        f"Fraud row ({fraud_prob:.4f}) should score higher than normal row ({normal_prob:.4f})"
    )


# ── Batch predictions ─────────────────────────────────────────────────────────

def test_batch_prediction_shape(tuned_model, test_sample):
    X, y = test_sample
    preds = tuned_model.predict(X)
    assert preds.shape == (len(X),)


def test_batch_proba_shape(tuned_model, test_sample):
    X, y = test_sample
    proba = tuned_model.predict_proba(X)
    assert proba.shape == (len(X), 2)


def test_batch_proba_all_in_unit_interval(tuned_model, test_sample):
    X, _ = test_sample
    proba = tuned_model.predict_proba(X)
    assert (proba >= 0).all() and (proba <= 1).all()


def test_batch_predictions_all_binary(tuned_model, test_sample):
    X, _ = test_sample
    preds = tuned_model.predict(X)
    assert set(preds).issubset({0, 1})


def test_model_detects_some_fraud(tuned_model, test_sample):
    """Model should flag at least one fraud in a 10-fraud sample."""
    X, y = test_sample
    proba = tuned_model.predict_proba(X)[:, 1]
    fraud_probs = proba[y.values == 1]
    assert (fraud_probs > 0.3).any(), (
        "Model flagged none of the 10 fraud rows above 0.3 — likely broken"
    )


# ── AUC sanity ────────────────────────────────────────────────────────────────

def test_model_auc_above_baseline(tuned_model, test_sample):
    from sklearn.metrics import roc_auc_score
    X, y = test_sample
    proba = tuned_model.predict_proba(X)[:, 1]
    auc = roc_auc_score(y, proba)
    assert auc > 0.85, f"AUC {auc:.4f} is below expected minimum of 0.85"
