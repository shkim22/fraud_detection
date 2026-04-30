from __future__ import annotations

import pathlib
import sys

import pandas as pd

# Support both `python src/data/quality.py` and `python -m src.data.quality`
if __package__ is None:
    sys.path.insert(0, str(pathlib.Path(__file__).parents[2]))

from src.data.loader import load

_NON_NEGATIVE_HINTS = ("count", "cnt", "qty", "quantity", "num_", "n_", "age", "duration", "size")
_RATE_HINTS = ("rate", "pct", "percent", "ratio", "proportion")
_TARGET_HINTS = ("class", "target", "label", "y")


def check_data_quality(
    df: pd.DataFrame,
    required_columns: dict[str, str] | None = None,
    target_column: str | None = None,
) -> dict:
    failures: list[str] = []
    warnings: list[str] = []

    null_counts = df.isnull().sum()
    statistics: dict = {
        "total_rows": len(df),
        "total_columns": len(df.columns),
        "total_nulls_by_column": null_counts[null_counts > 0].to_dict(),
    }

    # 1. Schema validation
    if required_columns:
        for col, expected_dtype in required_columns.items():
            if col not in df.columns:
                failures.append(f"Schema: missing required column '{col}'")
            elif not pd.api.types.is_dtype_equal(df[col].dtype, expected_dtype):
                failures.append(
                    f"Schema: '{col}' is {df[col].dtype}, expected {expected_dtype}"
                )

    # 2. Row count
    if len(df) < 100:
        failures.append(f"Row count: only {len(df):,} rows (minimum 100 required)")
    elif len(df) < 1000:
        warnings.append(f"Row count: {len(df):,} rows is low (recommend >= 1,000 for reliable modeling)")

    # 3. Null rates
    null_rates = df.isnull().mean()
    for col, rate in null_rates.items():
        if rate > 0.5:
            failures.append(f"Nulls: '{col}' is {rate:.1%} null (critical threshold 50%)")
        elif rate > 0.2:
            warnings.append(f"Nulls: '{col}' is {rate:.1%} null (warning threshold 20%)")

    # 4. Value ranges
    for col in df.select_dtypes(include="number").columns:
        col_lower = col.lower()
        series = df[col].dropna()
        if series.empty:
            continue
        if any(h in col_lower for h in _NON_NEGATIVE_HINTS):
            n_neg = int((series < 0).sum())
            if n_neg:
                failures.append(
                    f"Range: '{col}' has {n_neg:,} negative values (expected non-negative)"
                )
        if any(h in col_lower for h in _RATE_HINTS):
            n_extreme = int((series.abs() > 10_000).sum())
            if n_extreme:
                failures.append(
                    f"Range: '{col}' has {n_extreme:,} values > 10,000 (possible erroneous percentage)"
                )

    # 5. Target distribution
    if target_column is not None:
        if target_column not in df.columns:
            failures.append(f"Target: column '{target_column}' not found")
        else:
            dist = df[target_column].value_counts(normalize=True).sort_index()
            statistics["target_class_distribution"] = {
                str(k): round(float(v), 4) for k, v in dist.items()
            }
            n_classes = len(dist)
            if n_classes < 2:
                failures.append(
                    f"Target: '{target_column}' has only {n_classes} class — cannot train a classifier"
                )
            else:
                min_pct = float(dist.min())
                if min_pct < 0.05:
                    warnings.append(
                        f"Target: '{target_column}' is imbalanced — "
                        f"minority class is {min_pct:.2%} of data (< 5%)"
                    )

    return {
        "success": len(failures) == 0,
        "failures": failures,
        "warnings": warnings,
        "statistics": statistics,
    }


def _print_report(result: dict) -> None:
    status = "PASSED" if result["success"] else "FAILED"
    print(f"Quality gate: {status}\n")

    if result["failures"]:
        print(f"Failures ({len(result['failures'])}):")
        for f in result["failures"]:
            print(f"  [FAIL] {f}")
        print()

    if result["warnings"]:
        print(f"Warnings ({len(result['warnings'])}):")
        for w in result["warnings"]:
            print(f"  [WARN] {w}")
        print()

    stats = result["statistics"]
    print("Statistics:")
    print(f"  rows     : {stats['total_rows']:,}")
    print(f"  columns  : {stats['total_columns']}")
    nulls = stats["total_nulls_by_column"]
    print(f"  nulls    : {sum(nulls.values()):,} total across {len(nulls)} column(s)")
    if "target_class_distribution" in stats:
        print("  target distribution:")
        for cls, pct in stats["target_class_distribution"].items():
            print(f"    class {cls}: {pct:.2%}")


if __name__ == "__main__":
    df = load()

    # Auto-detect required columns from the loaded schema
    required = {col: str(df[col].dtype) for col in df.columns}

    # Auto-detect target column
    target = next(
        (col for col in df.columns if col.lower() in _TARGET_HINTS),
        None,
    )

    result = check_data_quality(df, required_columns=required, target_column=target)
    _print_report(result)
