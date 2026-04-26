import pathlib
import sys

import numpy as np
import pandas as pd

if __package__ is None:
    sys.path.insert(0, str(pathlib.Path(__file__).parents[2]))

from src.data.loader import DATA_DIR

_CLEANED_PATH = DATA_DIR / "cleaned.csv"
_V_COLS = [f"V{i}" for i in range(1, 29)]
_TOP_FRAUD_COLS = ["V14", "V17", "V12"]  # highest |corr| with Class (from EDA)


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # -----------------------------------------------------------------------
    # Category 1: Domain features  (fraud-specific business logic)
    # -----------------------------------------------------------------------

    # Amount is right-skewed (most txns < $100, a few > $10k). Log-scaling
    # compresses the tail so distance-based and regularised models treat a
    # $10 → $20 jump the same as $1,000 → $2,000 proportionally.
    df["amount_log"] = np.log1p(df["Amount"])

    # Bots and fraud scripts tend to use round dollar amounts (e.g. $100.00,
    # $500.00) rather than the irregular totals seen in retail purchases.
    df["amount_rounded"] = (df["Amount"] % 1 < 0.01).astype(int)

    # Micro-transactions (< $5) are a classic card-testing pattern: a stolen
    # card is charged a small amount first to confirm it is active before a
    # larger fraudulent purchase is attempted.
    df["amount_small"] = (df["Amount"] < 5).astype(int)

    # Unusually large single charges deviate from typical cardholder behaviour
    # and are a common threshold for manual fraud review at card networks.
    df["amount_large"] = (df["Amount"] > 500).astype(int)

    # Hour-of-day from the Time offset (seconds since the first transaction).
    # The dataset covers ~48 hours, so modulo 86 400 gives the within-day hour.
    df["time_hour"] = (df["Time"] % 86_400 // 3_600).astype(int)

    # Transactions between 10 PM and 6 AM are under-represented for legitimate
    # cardholders but over-represented in fraud — fraudsters exploit the reduced
    # real-time monitoring that banks run overnight.
    df["is_nighttime"] = (
        (df["time_hour"] >= 22) | (df["time_hour"] < 6)
    ).astype(int)

    # -----------------------------------------------------------------------
    # Category 2: Statistical features  (row-wise summaries across V1–V28)
    # -----------------------------------------------------------------------

    # Mean PCA component value per transaction. PCA was zero-centred on the
    # training population, so a large non-zero row mean indicates the
    # transaction sits well away from typical behaviour in latent space.
    df["v_mean"] = df[_V_COLS].mean(axis=1)

    # Row-wise standard deviation across all 28 components. High spread means
    # the transaction is extreme in many PCA directions simultaneously, which
    # is unusual for legitimate cardholders.
    df["v_std"] = df[_V_COLS].std(axis=1)

    # The single most extreme (absolute) PCA component per row. Fraud
    # transactions often spike on one specific dimension even when others
    # look near-normal, so the max captures localised anomalies.
    df["v_abs_max"] = df[_V_COLS].abs().max(axis=1)

    # Composite signal from the three V features with the highest absolute
    # correlation with Class (V14, V17, V12 per EDA). Averaging reduces noise
    # while retaining the strongest directional fraud indicator.
    df["top_fraud_signal"] = df[_TOP_FRAUD_COLS].mean(axis=1)

    # -----------------------------------------------------------------------
    # Category 3: Interaction features  (joint effects stronger than parts)
    # -----------------------------------------------------------------------

    # V14 × V12: the two individually most predictive PCA features. Their
    # product is large only when *both* are simultaneously extreme — a more
    # specific fraud fingerprint than either alone.
    df["v14_x_v12"] = df["V14"] * df["V12"]

    # log(Amount) × V14: flags high-value transactions that *also* carry the
    # primary PCA fraud signal. A large charge with a suspicious latent profile
    # is a particularly high-risk combination.
    df["amount_log_x_v14"] = df["amount_log"] * df["V14"]

    # V17 × V12: captures a complementary fraud interaction orthogonal to
    # v14_x_v12, covering cases where V14 is unremarkable but V17 and V12
    # are jointly extreme.
    df["v17_x_v12"] = df["V17"] * df["V12"]

    return df


def select_features(
    df: pd.DataFrame,
    target_col: str = "Class",
    corr_threshold: float = 0.95,
    variance_threshold_pct: float = 0.01,
) -> tuple[list[str], pd.DataFrame]:
    """Remove redundant and near-constant features.

    Drops in two passes:
    1. High-correlation pass  — for each pair with |r| > corr_threshold,
       drop the second feature (keep the one that appears first in the column
       order, which is typically the original rather than the engineered copy).
    2. Low-variance pass      — drop any feature whose variance is below
       variance_threshold_pct * mean(all feature variances).

    The target column and any non-numeric columns are never dropped.

    Returns (selected_feature_names, reduced_dataframe).
    """
    numeric_cols = [
        c for c in df.select_dtypes(include="number").columns if c != target_col
    ]
    dropped: dict[str, str] = {}  # col -> reason

    # ------------------------------------------------------------------
    # Pass 1: correlation filter
    # ------------------------------------------------------------------
    corr = df[numeric_cols].corr().abs()
    # Upper triangle only — avoid comparing a column with itself
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))

    for col in upper.columns:
        if col in dropped:
            continue
        partners = upper.index[upper[col] > corr_threshold].tolist()
        for partner in partners:
            if partner not in dropped:
                dropped[partner] = f"corr({col}, {partner}) = {upper.loc[partner, col]:.3f} > {corr_threshold}"

    after_corr = [c for c in numeric_cols if c not in dropped]

    # ------------------------------------------------------------------
    # Pass 2: variance filter
    # ------------------------------------------------------------------
    variances = df[after_corr].var()
    # Use median rather than mean so one extreme-variance column (e.g. Time,
    # whose raw-seconds range inflates the mean by orders of magnitude) does
    # not push the cutoff so high that every other feature is discarded.
    cutoff = variance_threshold_pct * variances.median()

    for col in after_corr:
        if variances[col] < cutoff:
            dropped[col] = f"var={variances[col]:.6f} < threshold={cutoff:.6f} ({variance_threshold_pct*100:.1f}% of mean var)"

    selected = [c for c in numeric_cols if c not in dropped]

    # ------------------------------------------------------------------
    # Log
    # ------------------------------------------------------------------
    print(f"Feature selection: {len(numeric_cols)} → {len(selected)} features")
    print(f"  Dropped {len(dropped)} feature(s):\n")
    for col, reason in dropped.items():
        print(f"  [DROP] {col:<25}  {reason}")
    if not dropped:
        print("  (none dropped)")

    non_feature_cols = [c for c in df.columns if c not in numeric_cols]
    return selected, df[non_feature_cols + selected]


if __name__ == "__main__":
    df = pd.read_csv(_CLEANED_PATH)
    print(f"Input : {df.shape[0]:,} rows, {df.shape[1]} columns")

    out = create_features(df)
    new_cols = [c for c in out.columns if c not in df.columns]

    print(f"Output: {out.shape[0]:,} rows, {out.shape[1]} columns")
    print(f"\n{len(new_cols)} new features created:\n")
    print(f"  {'feature':<25}  {'mean':>9}  {'std':>9}  {'min':>9}  {'max':>9}")
    print(f"  {'-'*25}  {'-'*9}  {'-'*9}  {'-'*9}  {'-'*9}")
    for col in new_cols:
        s = out[col]
        print(f"  {col:<25}  {s.mean():>9.4f}  {s.std():>9.4f}  {s.min():>9.4f}  {s.max():>9.4f}")

    print()
    selected, reduced = select_features(out)
    print(f"\nFinal shape: {reduced.shape[0]:,} rows, {reduced.shape[1]} columns")
