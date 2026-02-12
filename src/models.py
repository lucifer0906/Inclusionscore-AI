"""Model definitions, training helpers, and artifact loading."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from xgboost import XGBClassifier

from src.config import MODEL_DIR, MODEL_NAME, RANDOM_SEED


# ── Evaluation helper ────────────────────────────────────────────────────

def evaluate(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    *,
    label: str = "",
) -> dict[str, float]:
    """Compute classification metrics for a fitted model.

    Parameters
    ----------
    model
        Fitted estimator with ``predict`` and ``predict_proba`` methods.
    X : pd.DataFrame
        Feature matrix.
    y : pd.Series
        True labels.
    label : str
        Optional label printed in the console summary.

    Returns
    -------
    dict[str, float]
        Dictionary of metric name -> value.
    """
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]

    metrics = {
        "auc_roc": roc_auc_score(y, y_proba),
        "precision": precision_score(y, y_pred, zero_division=0),
        "recall": recall_score(y, y_pred, zero_division=0),
        "f1": f1_score(y, y_pred, zero_division=0),
        "accuracy": accuracy_score(y, y_pred),
        "brier_score": brier_score_loss(y, y_proba),
    }

    if label:
        print(f"\n{'=' * 40}")
        print(f"  {label}")
        print(f"{'=' * 40}")
        for k, v in metrics.items():
            print(f"  {k:>12s}: {v:.4f}")
        print()

    return metrics


def find_optimal_threshold(
    model,
    X: pd.DataFrame,
    y: pd.Series,
) -> dict[str, float]:
    """Find optimal classification threshold by maximizing F1-score.

    Returns
    -------
    dict
        Contains ``optimal_threshold``, ``f1_at_threshold``,
        ``precision_at_threshold``, ``recall_at_threshold``.
    """
    y_proba = model.predict_proba(X)[:, 1]
    precisions, recalls, thresholds = precision_recall_curve(y, y_proba)
    # F1 = 2 * (precision * recall) / (precision + recall)
    f1s = 2 * (precisions[:-1] * recalls[:-1]) / (precisions[:-1] + recalls[:-1] + 1e-12)
    best_idx = int(np.argmax(f1s))
    return {
        "optimal_threshold": float(thresholds[best_idx]),
        "f1_at_threshold": float(f1s[best_idx]),
        "precision_at_threshold": float(precisions[best_idx]),
        "recall_at_threshold": float(recalls[best_idx]),
    }


# ── Baseline: Logistic Regression ────────────────────────────────────────

def train_baseline(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame | None = None,
    y_val: pd.Series | None = None,
) -> tuple[LogisticRegression, dict[str, float]]:
    """Train a logistic regression baseline with class-weight balancing.

    Parameters
    ----------
    X_train, y_train
        Training data.
    X_val, y_val
        Optional validation data for reporting metrics.

    Returns
    -------
    tuple[LogisticRegression, dict]
        Fitted model and validation (or training) metrics.
    """
    lr = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=RANDOM_SEED,
        solver="lbfgs",
    )
    lr.fit(X_train, y_train)

    eval_X = X_val if X_val is not None else X_train
    eval_y = y_val if y_val is not None else y_train
    metrics = evaluate(lr, eval_X, eval_y, label="Logistic Regression (baseline)")

    return lr, metrics


# ── XGBoost ──────────────────────────────────────────────────────────────

_DEFAULT_XGB_PARAMS: dict[str, Any] = {
    "n_estimators": 500,
    "max_depth": 5,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "scale_pos_weight": 14,  # ~ratio of negatives/positives (1:14)
    "eval_metric": "auc",
    "random_state": RANDOM_SEED,
    "n_jobs": -1,
}


def train_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    config: dict[str, Any] | None = None,
) -> tuple[XGBClassifier, dict[str, float]]:
    """Train an XGBoost classifier with early stopping.

    Parameters
    ----------
    X_train, y_train
        Training data.
    X_val, y_val
        Validation data used for early stopping and metric reporting.
    config : dict, optional
        Override default hyper-parameters.

    Returns
    -------
    tuple[XGBClassifier, dict]
        Fitted model and validation metrics.
    """
    params = {**_DEFAULT_XGB_PARAMS, **(config or {})}

    xgb = XGBClassifier(**params)
    xgb.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=50,
    )

    metrics = evaluate(xgb, X_val, y_val, label="XGBoost (val)")

    return xgb, metrics


# ── Saving / loading ─────────────────────────────────────────────────────

def save_model(
    model,
    metrics: dict[str, float],
    name: str = MODEL_NAME,
    output_dir: Path | None = None,
    *,
    extra_meta: dict[str, Any] | None = None,
    X_train_rows: int = 0,
    X_val_rows: int = 0,
) -> Path:
    """Persist model artifact + metadata JSON side by side.

    Parameters
    ----------
    model
        Fitted estimator (must be joblib-serialisable).
    metrics : dict
        Evaluation metrics to store in metadata.
    name : str
        Base filename (without extension).
    output_dir : Path | None
        Target directory. Defaults to ``MODEL_DIR``.
    extra_meta : dict, optional
        Additional key-value pairs for the metadata file.
    X_train_rows, X_val_rows : int
        Row counts stored in metadata for reproducibility.

    Returns
    -------
    Path
        Path to the saved ``.joblib`` file.
    """
    output_dir = output_dir or MODEL_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = output_dir / f"{name}.joblib"
    meta_path = output_dir / f"{name}_metadata.json"

    joblib.dump(model, model_path)

    meta: dict[str, Any] = {
        "model_name": name,
        "dataset": "give_me_some_credit/cs-training.csv",
        "date": datetime.now(timezone.utc).isoformat(),
        "seed": RANDOM_SEED,
        "training_rows": X_train_rows,
        "validation_rows": X_val_rows,
        "metrics": {k: round(v, 6) for k, v in metrics.items()},
    }

    # Store XGBoost params if available
    if hasattr(model, "get_params"):
        params = model.get_params()
        # Keep only JSON-serialisable scalar params, convert NaN to None
        clean_params: dict[str, Any] = {}
        for k, v in params.items():
            if isinstance(v, float) and np.isnan(v):
                clean_params[k] = None
            elif isinstance(v, (int, float, str, bool, type(None))):
                clean_params[k] = v
        meta["params"] = clean_params

    if extra_meta:
        meta.update(extra_meta)

    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2, default=str)

    print(f"Model saved  -> {model_path}")
    print(f"Metadata     -> {meta_path}")
    return model_path


def load_model(path: str | Path):
    """Load a joblib-serialised model from disk.

    Parameters
    ----------
    path : str | Path
        Path to the ``.joblib`` file.

    Returns
    -------
    object
        The deserialised model.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")
    return joblib.load(path)


def load_model_metadata(model_path: str | Path) -> dict[str, Any]:
    """Load the metadata JSON that accompanies a model artifact."""
    model_path = Path(model_path)
    meta_path = model_path.with_name(
        model_path.stem + "_metadata.json"
    )
    if not meta_path.exists():
        return {}
    with open(meta_path) as f:
        return json.load(f)
