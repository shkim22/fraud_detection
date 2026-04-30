import sys
import pathlib

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).parents[1]))
from src.data.quality import check_data_quality

ROOT = pathlib.Path(__file__).parents[1]
V_COLS = [f"V{i}" for i in range(1, 29)]


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def cleaned_df():
    return pd.read_csv(ROOT / "data" / "cleaned.csv")


def _minimal_good_df(n=200):
    """Minimal valid credit-card-style dataframe that should pass the gate."""
    rng = np.random.default_rng(42)
    data = {col: rng.standard_normal(n) for col in V_COLS}
    data["Time"]   = rng.uniform(0, 172_800, n)
    data["Amount"] = rng.uniform(0.01, 5000, n)
    data["Class"]  = rng.integers(0, 2, n)
    return pd.DataFrame(data)


# ── Tests: quality gate passes on good data ───────────────────────────────────

def test_quality_gate_passes_on_cleaned_data(cleaned_df):
    result = check_data_quality(cleaned_df, target_column="Class")
    assert result["success"], (
        f"Quality gate failed on cleaned.csv.\nFailures: {result['failures']}"
    )


def test_quality_gate_returns_expected_keys(cleaned_df):
    result = check_data_quality(cleaned_df)
    assert set(result.keys()) == {"success", "failures", "warnings", "statistics"}


def test_quality_gate_no_failures_on_clean_data(cleaned_df):
    result = check_data_quality(cleaned_df, target_column="Class")
    assert result["failures"] == [], result["failures"]


def test_quality_gate_reports_statistics(cleaned_df):
    result = check_data_quality(cleaned_df, target_column="Class")
    stats = result["statistics"]
    assert stats["total_rows"] == len(cleaned_df)
    assert stats["total_columns"] == len(cleaned_df.columns)
    assert "target_class_distribution" in stats


def test_quality_gate_detects_target_imbalance(cleaned_df):
    # The real dataset is ~0.17% fraud — gate should warn, not fail
    result = check_data_quality(cleaned_df, target_column="Class")
    imbalance_warnings = [w for w in result["warnings"] if "imbalanced" in w.lower()]
    assert len(imbalance_warnings) == 1


def test_quality_gate_passes_on_minimal_good_df():
    result = check_data_quality(_minimal_good_df(), target_column="Class")
    assert result["success"]


# ── Tests: quality gate catches broken datasets ───────────────────────────────

def test_quality_gate_fails_on_too_few_rows():
    tiny = _minimal_good_df(n=50)   # below the 100-row minimum
    result = check_data_quality(tiny)
    assert not result["success"]
    assert any("Row count" in f for f in result["failures"])


def test_quality_gate_fails_on_missing_required_column():
    df = _minimal_good_df()
    df = df.drop(columns=["V1"])
    result = check_data_quality(df, required_columns={"V1": "float64"})
    assert not result["success"]
    assert any("V1" in f for f in result["failures"])


def test_quality_gate_fails_on_critical_null_rate():
    df = _minimal_good_df()
    # Inject >50% nulls into one column
    df.loc[: len(df) // 2, "Amount"] = np.nan
    result = check_data_quality(df)
    assert not result["success"]
    assert any("Amount" in f and "null" in f.lower() for f in result["failures"])


def test_quality_gate_warns_on_moderate_null_rate():
    df = _minimal_good_df(n=500)
    # Inject 25% nulls — warning threshold, not failure
    null_idx = df.sample(frac=0.25, random_state=0).index
    df.loc[null_idx, "Amount"] = np.nan
    result = check_data_quality(df)
    assert any("Amount" in w for w in result["warnings"])


def test_quality_gate_fails_when_target_has_one_class():
    df = _minimal_good_df()
    df["Class"] = 0   # only one class — untrainable
    result = check_data_quality(df, target_column="Class")
    assert not result["success"]
    assert any("class" in f.lower() for f in result["failures"])


def test_quality_gate_fails_when_required_schema_dtype_wrong():
    df = _minimal_good_df()
    df["Amount"] = df["Amount"].astype(str)   # wrong dtype
    result = check_data_quality(df, required_columns={"Amount": "float64"})
    assert not result["success"]
    assert any("Amount" in f for f in result["failures"])
