"""Hyperparameter tuning with Optuna for XGBoost.

Uses k-fold stratified cross-validation to find optimal parameters,
then saves the best configuration to ``models/best_params.json``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import optuna
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score

from src.config import MODEL_DIR, RANDOM_SEED

BEST_PARAMS_PATH = MODEL_DIR / "best_params.json"


def _objective(
    trial: optuna.Trial,
    X: np.ndarray,
    y: np.ndarray,
    n_folds: int = 5,
) -> float:
    """Optuna objective: 5-fold stratified CV AUC-ROC."""
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000, step=50),
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "scale_pos_weight": trial.suggest_float("scale_pos_weight", 10, 20),
        "eval_metric": "auc",
        "random_state": RANDOM_SEED,
        "n_jobs": -1,
        "verbosity": 0,
    }

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_SEED)
    aucs = []

    for train_idx, val_idx in skf.split(X, y):
        X_tr, X_va = X[train_idx], X[val_idx]
        y_tr, y_va = y[train_idx], y[val_idx]

        model = XGBClassifier(**params)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_va, y_va)],
            verbose=False,
        )
        proba = model.predict_proba(X_va)[:, 1]
        aucs.append(roc_auc_score(y_va, proba))

    return float(np.mean(aucs))


def tune_xgboost(
    X: np.ndarray,
    y: np.ndarray,
    *,
    n_trials: int = 50,
    n_folds: int = 5,
    output_path: Path | None = None,
) -> dict:
    """Run Optuna hyperparameter search for XGBoost.

    Parameters
    ----------
    X : np.ndarray
        Training features.
    y : np.ndarray
        Training labels.
    n_trials : int
        Number of Optuna trials.
    n_folds : int
        Cross-validation folds.
    output_path : Path | None
        Where to save best params JSON.

    Returns
    -------
    dict
        Best parameters and CV score.
    """
    output_path = output_path or BEST_PARAMS_PATH

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize")
    study.optimize(
        lambda trial: _objective(trial, X, y, n_folds),
        n_trials=n_trials,
        show_progress_bar=True,
    )

    best = study.best_params
    best["eval_metric"] = "auc"
    best["random_state"] = RANDOM_SEED
    best["n_jobs"] = -1

    result = {
        "best_params": best,
        "best_cv_auc": round(study.best_value, 6),
        "n_trials": n_trials,
        "n_folds": n_folds,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Best CV AUC: {study.best_value:.6f}")
    print(f"Best params saved to {output_path}")

    return result


def load_best_params(path: Path | None = None) -> dict | None:
    """Load tuned params from JSON, or return None if not found."""
    path = path or BEST_PARAMS_PATH
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return data.get("best_params")
