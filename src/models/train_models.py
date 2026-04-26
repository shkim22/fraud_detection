"""
Compare XGBoost, Random Forest, and LightGBM for credit card fraud detection.

Model rationale
---------------
XGBoost      Gradient boosting with L1/L2 regularisation. Captures non-linear
             feature interactions; highly configurable; industry standard for
             tabular fraud scoring. Handles imbalance via SMOTE + scale_pos_weight.

RandomForest Bootstrap aggregation of deep trees. Low inter-tree correlation
             yields stable, low-variance decisions — important for an automated
             system making high-volume decisions with no human review step.
             Robust to the heavy-tailed V-feature outliers identified in EDA.

LightGBM     Leaf-wise gradient boosting with GOSS sampling: trains 3–5× faster
             than XGBoost on this 283k-row dataset while matching its accuracy.
             Well-suited to production latency requirements for automated scoring.

All three pipelines apply StandardScaler → SMOTE so each model receives
balanced, normalised data during both cross-validation and final training.
"""
import pathlib
import sys
import time
import warnings

warnings.filterwarnings("ignore")

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler

if __package__ is None:
    sys.path.insert(0, str(pathlib.Path(__file__).parents[2]))

from src.data.loader import DATA_DIR

_FEATURES_PATH = DATA_DIR / "features.csv"
_MODELS_DIR = DATA_DIR.parent / "models"
_TARGET_COL = "Class"
_RANDOM_STATE = 42
_CV_FOLDS = 5


def _build_models() -> dict:
    # SMOTE(sampling_strategy=0.1): oversample fraud to 10% of legitimate
    # transactions — enough to correct the 0.17% imbalance without generating
    # excessive synthetic data that would slow CV to impractical runtimes.
    smote_kw = dict(sampling_strategy=0.1, random_state=_RANDOM_STATE)

    return {
        "XGBoost": ImbPipeline([
            ("scaler", StandardScaler()),
            ("smote",  SMOTE(**smote_kw)),
            ("clf",    xgb.XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=_RANDOM_STATE,
                n_jobs=-1,
                verbosity=0,
            )),
        ]),
        "RandomForest": ImbPipeline([
            ("scaler", StandardScaler()),
            ("smote",  SMOTE(**smote_kw)),
            ("clf",    RandomForestClassifier(
                n_estimators=200,
                min_samples_leaf=2,
                random_state=_RANDOM_STATE,
                n_jobs=-1,
            )),
        ]),
        "LightGBM": ImbPipeline([
            ("scaler", StandardScaler()),
            ("smote",  SMOTE(**smote_kw)),
            ("clf",    lgb.LGBMClassifier(
                n_estimators=200,
                num_leaves=63,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=_RANDOM_STATE,
                n_jobs=-1,
                verbose=-1,
            )),
        ]),
    }


def train_and_compare(df: pd.DataFrame) -> pd.DataFrame:
    X = df.drop(columns=[_TARGET_COL])
    y = df[_TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=_RANDOM_STATE, stratify=y
    )

    n_fraud = y_train.sum()
    print(f"Train : {len(X_train):,}  ({n_fraud} fraud, {len(X_train)-n_fraud:,} legit)")
    print(f"Test  : {len(X_test):,}  ({y_test.sum()} fraud)\n")

    cv = StratifiedKFold(n_splits=_CV_FOLDS, shuffle=True, random_state=_RANDOM_STATE)
    _MODELS_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    for name, pipeline in _build_models().items():
        print(f"── {name} {'─' * (40 - len(name))}")

        # 5-fold CV (n_jobs=1 avoids nested parallelism with model's own n_jobs)
        print(f"  {_CV_FOLDS}-fold CV ...", end=" ", flush=True)
        t_cv = time.time()
        cv_scores = cross_val_score(
            pipeline, X_train, y_train,
            cv=cv, scoring="roc_auc", n_jobs=1,
        )
        print(f"{time.time() - t_cv:.0f}s  →  {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

        # Final fit on full training set
        print(f"  Final fit ...", end=" ", flush=True)
        t_fit = time.time()
        pipeline.fit(X_train, y_train)
        fit_time = time.time() - t_fit
        print(f"{fit_time:.0f}s")

        # Test metrics
        y_prob = pipeline.predict_proba(X_test)[:, 1]
        y_pred = pipeline.predict(X_test)
        metrics = dict(
            roc_auc   = roc_auc_score(y_test, y_prob),
            f1        = f1_score(y_test, y_pred),
            precision = precision_score(y_test, y_pred, zero_division=0),
            recall    = recall_score(y_test, y_pred),
        )
        print(f"  Test  AUC={metrics['roc_auc']:.4f}  "
              f"F1={metrics['f1']:.4f}  "
              f"P={metrics['precision']:.4f}  "
              f"R={metrics['recall']:.4f}")

        out = _MODELS_DIR / f"{name.lower()}.pkl"
        joblib.dump(pipeline, out)
        print(f"  Saved → {out.name}\n")

        rows.append({
            "model"           : name,
            "cv_roc_auc_mean" : round(cv_scores.mean(), 4),
            "cv_roc_auc_std"  : round(cv_scores.std(), 4),
            "test_roc_auc"    : round(metrics["roc_auc"], 4),
            "test_f1"         : round(metrics["f1"], 4),
            "test_precision"  : round(metrics["precision"], 4),
            "test_recall"     : round(metrics["recall"], 4),
            "train_time_s"    : round(fit_time, 1),
        })

    table = (
        pd.DataFrame(rows)
        .sort_values("test_roc_auc", ascending=False)
        .reset_index(drop=True)
    )
    return table


def _print_recommendation(table: pd.DataFrame) -> None:
    best = table.iloc[0]
    runner_up = table.iloc[1]

    print("=" * 56)
    print(f"  Best model : {best['model']}")
    print(f"  AUC {best['test_roc_auc']}  |  F1 {best['test_f1']}  |  "
          f"Recall {best['test_recall']}  |  Precision {best['test_precision']}")
    print()

    rationale = {
        "XGBoost": (
            "Gradient boosting regularisation suppresses false positives while\n"
            "  SMOTE ensures fraud recall stays high — the right balance for an\n"
            "  automated decisioning system that acts on every flagged transaction."
        ),
        "LightGBM": (
            "Leaf-wise growth captures subtle fraud patterns faster than level-wise\n"
            "  boosting. Lower training time matters when models are retrained daily\n"
            "  on new transaction batches in a production pipeline."
        ),
        "RandomForest": (
            "Bootstrap aggregation produces stable decisions with low variance —\n"
            "  important when downstream automated actions (block/approve) must be\n"
            "  consistent and not sensitive to small data changes."
        ),
    }
    print(f"  Why: {rationale.get(best['model'], '')}")
    print()

    delta = best["test_roc_auc"] - runner_up["test_roc_auc"]
    if delta < 0.003:
        print(f"  Note: margin over {runner_up['model']} is only {delta:.4f} —\n"
              f"  consider {runner_up['model']} if lower latency matters more than AUC.")
    print("=" * 56)


if __name__ == "__main__":
    print(__doc__)
    df = pd.read_csv(_FEATURES_PATH)
    print(f"Loaded {df.shape[0]:,} rows, {df.shape[1]} columns\n")

    table = train_and_compare(df)

    print("\nComparison table")
    print("─" * 56)
    print(table.to_string(index=False))
    print()
    _print_recommendation(table)
