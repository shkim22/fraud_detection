"""
Hyperparameter tuning pipeline for credit card fraud detection.

Workflow
--------
1. Load data/features.csv and split 80/20 (stratified).
2. Quick 3-fold CV on default params to select the best model type
   among XGBoost, LightGBM, and RandomForest.
3. Optuna: 30 trials of 5-fold CV (ROC-AUC) on the winning model type.
4. Save best trial hyperparameters to models/best_params.json.
5. Train final pipeline on full training set with best params.
6. Evaluate on held-out test set and print metrics.
7. Save tuned model to models/tuned_model.pkl.
"""
import json
import logging
import pathlib
import sys
import warnings

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

import joblib
import lightgbm as lgb
import numpy as np
import optuna
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
_N_TRIALS = 30
_CV_FOLDS = 5
_SMOTE_KW = dict(sampling_strategy=0.1, random_state=_RANDOM_STATE)


# ---------------------------------------------------------------------------
# Pipeline factories
# ---------------------------------------------------------------------------

def _lgbm_pipeline(params: dict) -> ImbPipeline:
    return ImbPipeline([
        ("scaler", StandardScaler()),
        ("smote",  SMOTE(**_SMOTE_KW)),
        ("clf",    lgb.LGBMClassifier(
            random_state=_RANDOM_STATE, n_jobs=-1, verbose=-1, **params
        )),
    ])


def _xgb_pipeline(params: dict) -> ImbPipeline:
    return ImbPipeline([
        ("scaler", StandardScaler()),
        ("smote",  SMOTE(**_SMOTE_KW)),
        ("clf",    xgb.XGBClassifier(
            random_state=_RANDOM_STATE, n_jobs=-1, verbosity=0,
            eval_metric="logloss", **params
        )),
    ])


def _rf_pipeline(params: dict) -> ImbPipeline:
    return ImbPipeline([
        ("scaler", StandardScaler()),
        ("smote",  SMOTE(**_SMOTE_KW)),
        ("clf",    RandomForestClassifier(
            random_state=_RANDOM_STATE, n_jobs=-1, **params
        )),
    ])


_DEFAULT_PARAMS = {
    "LightGBM":    dict(n_estimators=200, num_leaves=63, learning_rate=0.1,
                        subsample=0.8, colsample_bytree=0.8),
    "XGBoost":     dict(n_estimators=200, max_depth=6, learning_rate=0.1,
                        subsample=0.8, colsample_bytree=0.8),
    "RandomForest": dict(n_estimators=200, min_samples_leaf=2),
}

_FACTORIES = {
    "LightGBM":    _lgbm_pipeline,
    "XGBoost":     _xgb_pipeline,
    "RandomForest": _rf_pipeline,
}


# ---------------------------------------------------------------------------
# Model selection
# ---------------------------------------------------------------------------

def _select_best_model(X_train: pd.DataFrame, y_train: pd.Series) -> str:
    """Run a quick 3-fold CV on default params; return the highest-AUC model name."""
    log.info("── Model selection (3-fold CV on defaults) ──────────────────")
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=_RANDOM_STATE)
    scores = {}
    for name, factory in _FACTORIES.items():
        pipeline = factory(_DEFAULT_PARAMS[name])
        cv_auc = cross_val_score(
            pipeline, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=1
        ).mean()
        scores[name] = cv_auc
        log.info(f"  {name:<14} ROC-AUC = {cv_auc:.4f}")

    best = max(scores, key=scores.__getitem__)
    log.info(f"\n  Selected → {best} (AUC {scores[best]:.4f})\n")
    return best


# ---------------------------------------------------------------------------
# Optuna objective builders
# ---------------------------------------------------------------------------

def _lgbm_objective(trial, X, y, cv):
    params = dict(
        n_estimators      = trial.suggest_int("n_estimators", 100, 600),
        num_leaves        = trial.suggest_int("num_leaves", 20, 150),
        max_depth         = trial.suggest_int("max_depth", 3, 12),
        learning_rate     = trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        subsample         = trial.suggest_float("subsample", 0.5, 1.0),
        colsample_bytree  = trial.suggest_float("colsample_bytree", 0.5, 1.0),
        min_child_samples = trial.suggest_int("min_child_samples", 5, 100),
        reg_alpha         = trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        reg_lambda        = trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
    )
    score = cross_val_score(
        _lgbm_pipeline(params), X, y, cv=cv, scoring="roc_auc", n_jobs=1
    ).mean()
    log.info(f"  trial {trial.number:>3}  AUC={score:.4f}  {params}")
    return score


def _xgb_objective(trial, X, y, cv):
    params = dict(
        n_estimators    = trial.suggest_int("n_estimators", 100, 600),
        max_depth       = trial.suggest_int("max_depth", 3, 10),
        learning_rate   = trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        subsample       = trial.suggest_float("subsample", 0.5, 1.0),
        colsample_bytree= trial.suggest_float("colsample_bytree", 0.5, 1.0),
        min_child_weight= trial.suggest_int("min_child_weight", 1, 20),
        gamma           = trial.suggest_float("gamma", 0.0, 5.0),
        reg_alpha       = trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        reg_lambda      = trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
    )
    score = cross_val_score(
        _xgb_pipeline(params), X, y, cv=cv, scoring="roc_auc", n_jobs=1
    ).mean()
    log.info(f"  trial {trial.number:>3}  AUC={score:.4f}  {params}")
    return score


def _rf_objective(trial, X, y, cv):
    params = dict(
        n_estimators    = trial.suggest_int("n_estimators", 100, 600),
        max_depth       = trial.suggest_int("max_depth", 5, 30),
        min_samples_split = trial.suggest_int("min_samples_split", 2, 20),
        min_samples_leaf  = trial.suggest_int("min_samples_leaf", 1, 20),
        max_features    = trial.suggest_categorical("max_features", ["sqrt", "log2", 0.5]),
    )
    score = cross_val_score(
        _rf_pipeline(params), X, y, cv=cv, scoring="roc_auc", n_jobs=1
    ).mean()
    log.info(f"  trial {trial.number:>3}  AUC={score:.4f}  {params}")
    return score


_OBJECTIVES = {
    "LightGBM":    _lgbm_objective,
    "XGBoost":     _xgb_objective,
    "RandomForest": _rf_objective,
}


# ---------------------------------------------------------------------------
# Main tuning function
# ---------------------------------------------------------------------------

def tune(features_path: pathlib.Path = _FEATURES_PATH) -> dict:
    df = pd.read_csv(features_path)
    log.info(f"Loaded {df.shape[0]:,} rows, {df.shape[1]} columns\n")

    X = df.drop(columns=[_TARGET_COL])
    y = df[_TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=_RANDOM_STATE, stratify=y
    )
    log.info(f"Train : {len(X_train):,}  ({y_train.sum()} fraud)")
    log.info(f"Test  : {len(X_test):,}  ({y_test.sum()} fraud)\n")

    best_model_name = _select_best_model(X_train, y_train)

    # Suppress Optuna's own per-trial output; we log manually in the objective.
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    cv = StratifiedKFold(n_splits=_CV_FOLDS, shuffle=True, random_state=_RANDOM_STATE)
    objective_fn = _OBJECTIVES[best_model_name]

    log.info(f"── Optuna tuning: {best_model_name} ({_N_TRIALS} trials, "
             f"{_CV_FOLDS}-fold CV) ──────────")
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=_RANDOM_STATE),
    )
    study.optimize(
        lambda trial: objective_fn(trial, X_train, y_train, cv),
        n_trials=_N_TRIALS,
        show_progress_bar=False,
    )

    best_params = study.best_trial.params
    best_cv_auc = study.best_trial.value
    log.info(f"\n  Best trial AUC : {best_cv_auc:.4f}")
    log.info(f"  Best params    : {best_params}\n")

    # Persist best hyperparameters
    _MODELS_DIR.mkdir(parents=True, exist_ok=True)
    params_path = _MODELS_DIR / "best_params.json"
    payload = {"model": best_model_name, "cv_roc_auc": best_cv_auc, **best_params}
    params_path.write_text(json.dumps(payload, indent=2))
    log.info(f"Best params saved → {params_path}\n")

    # Final model: train on full training set with best params
    log.info("── Final model training on full training set ────────────────")
    final_pipeline = _FACTORIES[best_model_name](best_params)
    final_pipeline.fit(X_train, y_train)

    # Evaluate on held-out test set
    y_prob = final_pipeline.predict_proba(X_test)[:, 1]
    y_pred = final_pipeline.predict(X_test)
    metrics = dict(
        roc_auc   = roc_auc_score(y_test, y_prob),
        f1        = f1_score(y_test, y_pred, zero_division=0),
        precision = precision_score(y_test, y_pred, zero_division=0),
        recall    = recall_score(y_test, y_pred, zero_division=0),
    )

    log.info("\nTest-set metrics:")
    log.info(f"  {'ROC-AUC':<12} {metrics['roc_auc']:.4f}")
    log.info(f"  {'F1':<12} {metrics['f1']:.4f}")
    log.info(f"  {'Precision':<12} {metrics['precision']:.4f}")
    log.info(f"  {'Recall':<12} {metrics['recall']:.4f}")

    model_path = _MODELS_DIR / "tuned_model.pkl"
    joblib.dump(final_pipeline, model_path)
    log.info(f"\nTuned model saved → {model_path}")

    return {"model": best_model_name, "best_params": best_params, "metrics": metrics}


if __name__ == "__main__":
    tune()
