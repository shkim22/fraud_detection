import pathlib
import sys

import pandas as pd

if __package__ is None:
    sys.path.insert(0, str(pathlib.Path(__file__).parents[2]))

from src.data.loader import load, DATA_DIR
from src.data.quality import check_data_quality, _print_report, _TARGET_HINTS

_OUTPUT_PATH = DATA_DIR / "cleaned.csv"

# Heuristic: a column whose name contains any of these is treated as a time index,
# which switches null handling from row-drop to forward-fill.
_TIME_HINTS = ("time", "date", "timestamp", "datetime", "period", "step")


def _is_time_series(df: pd.DataFrame) -> bool:
    return any(h in col.lower() for col in df.columns for h in _TIME_HINTS)


def _detect_target(df: pd.DataFrame) -> str | None:
    return next((col for col in df.columns if col.lower() in _TARGET_HINTS), None)


def clean_data(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    df = df.copy()
    target_col = _detect_target(df)

    # --- drop columns with > 50% nulls first (before any row logic) ---
    null_rates = df.isnull().mean()
    high_null_cols = null_rates[null_rates > 0.5].index.tolist()
    if high_null_cols:
        print(f"Dropping {len(high_null_cols)} column(s) with > 50% nulls: {high_null_cols}")
        df = df.drop(columns=high_null_cols)

    # --- handle target nulls (always drop rows where target is missing) ---
    if target_col and target_col in df.columns:
        n_before = len(df)
        df = df.dropna(subset=[target_col])
        dropped = n_before - len(df)
        if dropped:
            print(f"Dropped {dropped:,} row(s) with null target ('{target_col}')")

    # --- handle remaining nulls ---
    non_target = [c for c in df.columns if c != target_col]
    if _is_time_series(df):
        print("Time-series detected — forward-filling remaining nulls")
        df[non_target] = df[non_target].ffill()
        # any still-null rows at the very start (nothing to forward-fill from) → drop
        n_before = len(df)
        df = df.dropna(subset=non_target)
        dropped = n_before - len(df)
        if dropped:
            print(f"Dropped {dropped:,} leading row(s) that could not be forward-filled")
    else:
        n_before = len(df)
        df = df.dropna(subset=non_target)
        dropped = n_before - len(df)
        if dropped:
            print(f"Dropped {dropped:,} row(s) with remaining nulls")

    # --- remove exact duplicates ---
    n_before = len(df)
    df = df.drop_duplicates(keep="first")
    dropped = n_before - len(df)
    if dropped:
        print(f"Removed {dropped:,} duplicate row(s)")

    # --- dtype coercion ---
    for col in df.columns:
        if col == target_col:
            continue
        if pd.api.types.is_object_dtype(df[col]):
            # try numeric first; leave as string if it fails
            coerced = pd.to_numeric(df[col], errors="coerce")
            if coerced.isnull().mean() < 0.05:
                df[col] = coerced
            else:
                df[col] = df[col].astype(str)
        elif pd.api.types.is_bool_dtype(df[col]):
            df[col] = df[col].astype(int)

    # --- save ---
    df.to_csv(_OUTPUT_PATH, index=False)
    print(f"Saved cleaned data to {_OUTPUT_PATH}")

    # --- quality gate on cleaned data ---
    required = {col: str(df[col].dtype) for col in df.columns}
    quality = check_data_quality(df, required_columns=required, target_column=target_col)

    return df, quality


if __name__ == "__main__":
    raw = load()
    print(f"Raw data: {len(raw):,} rows, {raw.shape[1]} columns\n")

    cleaned, quality = clean_data(raw)

    print(f"\nAfter cleaning: {len(cleaned):,} rows, {cleaned.shape[1]} columns")
    print(f"Rows removed  : {len(raw) - len(cleaned):,}\n")

    _print_report(quality)
