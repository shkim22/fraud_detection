import sys
import pathlib

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).parents[1]))
from src.features.engineering import create_features, select_features

# 13 engineered features added by create_features:
ENGINEERED_COLS = [
    "amount_log", "amount_rounded", "amount_small", "amount_large",
    "time_hour", "is_nighttime",
    "v_mean", "v_std", "v_abs_max", "top_fraud_signal",
    "v14_x_v12", "amount_log_x_v14", "v17_x_v12",
]
V_COLS = [f"V{i}" for i in range(1, 29)]
INPUT_COLS = ["Time", "Amount", "Class"] + V_COLS


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def sample_df():
    """100-row sample of cleaned.csv, loaded once for the whole module."""
    root = pathlib.Path(__file__).parents[1]
    return pd.read_csv(root / "data" / "cleaned.csv").head(100)


@pytest.fixture(scope="module")
def engineered_df(sample_df):
    return create_features(sample_df)


def _make_df(n=50, seed=0):
    rng = np.random.default_rng(seed)
    data = {col: rng.standard_normal(n) for col in V_COLS}
    data["Time"]   = rng.uniform(0, 172_800, n)
    data["Amount"] = rng.uniform(0.01, 5000, n)
    data["Class"]  = rng.integers(0, 2, n)
    return pd.DataFrame(data)


# ── Column count ──────────────────────────────────────────────────────────────

def test_create_features_adds_expected_column_count(sample_df, engineered_df):
    expected = len(sample_df.columns) + len(ENGINEERED_COLS)
    assert engineered_df.shape[1] == expected, (
        f"Expected {expected} columns, got {engineered_df.shape[1]}"
    )


def test_all_engineered_columns_present(engineered_df):
    for col in ENGINEERED_COLS:
        assert col in engineered_df.columns, f"Missing column: {col}"


def test_row_count_unchanged(sample_df, engineered_df):
    assert len(engineered_df) == len(sample_df)


# ── No NaNs ───────────────────────────────────────────────────────────────────

def test_no_nan_in_output(engineered_df):
    null_counts = engineered_df.isnull().sum()
    nulls = null_counts[null_counts > 0]
    assert nulls.empty, f"NaNs found after feature engineering:\n{nulls}"


def test_no_nan_on_synthetic_input():
    out = create_features(_make_df())
    assert not out.isnull().values.any()


# ── Feature ranges and correctness ───────────────────────────────────────────

def test_amount_log_non_negative(engineered_df):
    assert (engineered_df["amount_log"] >= 0).all()


def test_amount_log_equals_log1p_amount(engineered_df):
    expected = np.log1p(engineered_df["Amount"])
    pd.testing.assert_series_equal(
        engineered_df["amount_log"].reset_index(drop=True),
        expected.reset_index(drop=True),
        check_names=False,
    )


def test_amount_rounded_is_binary(engineered_df):
    assert engineered_df["amount_rounded"].isin([0, 1]).all()


def test_amount_small_is_binary(engineered_df):
    assert engineered_df["amount_small"].isin([0, 1]).all()


def test_amount_large_is_binary(engineered_df):
    assert engineered_df["amount_large"].isin([0, 1]).all()


def test_is_nighttime_is_binary(engineered_df):
    assert engineered_df["is_nighttime"].isin([0, 1]).all()


def test_time_hour_range(engineered_df):
    assert engineered_df["time_hour"].between(0, 23).all()


def test_amount_small_flag_correct():
    df = pd.DataFrame({
        **{col: [0.0] for col in V_COLS},
        "Time": [0.0], "Amount": [3.99], "Class": [0],
    })
    out = create_features(df)
    assert out["amount_small"].iloc[0] == 1


def test_amount_small_flag_not_set_for_large_amount():
    df = pd.DataFrame({
        **{col: [0.0] for col in V_COLS},
        "Time": [0.0], "Amount": [150.0], "Class": [0],
    })
    out = create_features(df)
    assert out["amount_small"].iloc[0] == 0


def test_amount_large_flag_correct():
    df = pd.DataFrame({
        **{col: [0.0] for col in V_COLS},
        "Time": [0.0], "Amount": [999.0], "Class": [0],
    })
    out = create_features(df)
    assert out["amount_large"].iloc[0] == 1


def test_is_nighttime_set_at_midnight():
    df = pd.DataFrame({
        **{col: [0.0] for col in V_COLS},
        "Time": [0.0], "Amount": [50.0], "Class": [0],   # hour 0 → nighttime
    })
    out = create_features(df)
    assert out["is_nighttime"].iloc[0] == 1


def test_is_nighttime_not_set_at_noon():
    df = pd.DataFrame({
        **{col: [0.0] for col in V_COLS},
        "Time": [12 * 3600.0], "Amount": [50.0], "Class": [0],
    })
    out = create_features(df)
    assert out["is_nighttime"].iloc[0] == 0


def test_v14_x_v12_equals_product(engineered_df):
    expected = (engineered_df["V14"] * engineered_df["V12"]).values
    actual   = engineered_df["v14_x_v12"].values
    np.testing.assert_array_almost_equal(actual, expected)


def test_top_fraud_signal_equals_mean_of_v14_v17_v12(engineered_df):
    expected = engineered_df[["V14", "V17", "V12"]].mean(axis=1).values
    actual   = engineered_df["top_fraud_signal"].values
    np.testing.assert_array_almost_equal(actual, expected)


def test_v_mean_within_reasonable_range(engineered_df):
    # PCA components are standardised; row mean should stay within ±5
    assert engineered_df["v_mean"].abs().max() < 5.0


def test_v_abs_max_non_negative(engineered_df):
    assert (engineered_df["v_abs_max"] >= 0).all()


# ── select_features ───────────────────────────────────────────────────────────

def test_select_features_returns_fewer_or_equal_columns(engineered_df):
    selected, reduced = select_features(engineered_df)
    assert len(selected) <= engineered_df.shape[1] - 1   # -1 for target


def test_select_features_never_drops_target(engineered_df):
    selected, reduced = select_features(engineered_df, target_col="Class")
    assert "Class" in reduced.columns


def test_select_features_output_has_no_nan(engineered_df):
    _, reduced = select_features(engineered_df)
    assert not reduced.isnull().values.any()
