"""SHAP-based explainability utilities.

Provides functions to compute global SHAP summary plots and per-sample
local explanations for any tree-based model supported by SHAP's
TreeExplainer.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for saving plots
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from src.config import MODEL_DIR, REPORTS_DIR
from src.models import load_model
from src.features import load_processed

EXPLAINABILITY_DIR = REPORTS_DIR / "explainability"


def _load_model_and_data(
    model_path: str | Path,
    data_path: str | Path,
):
    """Load model and test data, returning (model, X, y)."""
    model = load_model(model_path)

    if str(data_path).endswith(".parquet"):
        df = pd.read_parquet(data_path)
    else:
        df = pd.read_csv(data_path)

    # Separate target if present
    if "SeriousDlqin2yrs" in df.columns:
        y = df.pop("SeriousDlqin2yrs")
    else:
        y = None

    return model, df, y


def compute_shap_summary(
    model_path: str | Path,
    data_path: str | Path,
    *,
    max_samples: int = 2000,
    output_dir: Path | None = None,
) -> np.ndarray:
    """Compute SHAP values and save a global summary plot.

    Parameters
    ----------
    model_path : str | Path
        Path to the saved model (``.joblib``).
    data_path : str | Path
        Path to the test data (Parquet or CSV).
    max_samples : int
        Max rows to use for SHAP computation (for speed).
    output_dir : Path | None
        Where to save plot PNGs. Defaults to ``reports/explainability/``.

    Returns
    -------
    np.ndarray
        SHAP values array.
    """
    output_dir = output_dir or EXPLAINABILITY_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    model, X, _ = _load_model_and_data(model_path, data_path)

    # Subsample for speed
    if len(X) > max_samples:
        X = X.sample(max_samples, random_state=42)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # 1. Summary bar plot (global feature importance)
    plt.figure(figsize=(10, 7))
    shap.summary_plot(shap_values, X, plot_type="bar", show=False)
    plt.title("SHAP Global Feature Importance")
    plt.tight_layout()
    plt.savefig(output_dir / "shap_summary.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {output_dir / 'shap_summary.png'}")

    # 2. Beeswarm / dot summary plot (shows direction of effect)
    plt.figure(figsize=(10, 7))
    shap.summary_plot(shap_values, X, show=False)
    plt.title("SHAP Summary (Beeswarm)")
    plt.tight_layout()
    plt.savefig(output_dir / "shap_beeswarm.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {output_dir / 'shap_beeswarm.png'}")

    return shap_values


def local_explanation(
    model_path: str | Path,
    data_path: str | Path,
    idx: int,
    *,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """Generate a local SHAP explanation for a single sample.

    Parameters
    ----------
    model_path : str | Path
        Path to the saved model.
    data_path : str | Path
        Path to the dataset.
    idx : int
        Row index within the dataset to explain.
    output_dir : Path | None
        Where to save the waterfall plot PNG.

    Returns
    -------
    dict[str, Any]
        Serialisable dict with ``feature``, ``value``, ``contribution``
        for each feature, plus the ``base_value`` and ``prediction``.
    """
    output_dir = output_dir or EXPLAINABILITY_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    model, X, _ = _load_model_and_data(model_path, data_path)
    row = X.iloc[[idx]]

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(row)
    base_value = float(explainer.expected_value)

    contribs = shap_values[0]
    feature_names = list(X.columns)

    # Build result dict
    top_features = sorted(
        zip(feature_names, row.values[0], contribs),
        key=lambda t: abs(t[2]),
        reverse=True,
    )
    result: dict[str, Any] = {
        "base_value": base_value,
        "prediction": float(model.predict_proba(row)[0, 1]),
        "top_features": [
            {"feature": f, "value": float(v), "contribution": float(c)}
            for f, v, c in top_features
        ],
    }

    # Save waterfall plot
    explanation = shap.Explanation(
        values=contribs,
        base_values=base_value,
        data=row.values[0],
        feature_names=feature_names,
    )
    plt.figure(figsize=(10, 6))
    shap.plots.waterfall(explanation, show=False)
    plt.title(f"Local Explanation (sample idx={idx})")
    plt.tight_layout()
    plt.savefig(
        output_dir / f"shap_local_{idx}.png", dpi=150, bbox_inches="tight"
    )
    plt.close()
    print(f"Saved {output_dir / f'shap_local_{idx}.png'}")

    return result


def generate_local_explanations(
    model_path: str | Path,
    data_path: str | Path,
    indices: list[int] | None = None,
    *,
    output_dir: Path | None = None,
) -> list[dict[str, Any]]:
    """Generate local explanations for multiple samples.

    Parameters
    ----------
    model_path, data_path
        Model and data paths.
    indices : list[int] | None
        Row indices to explain. Defaults to [0, 1, 2, 3, 4].
    output_dir : Path | None
        Output directory for plots.

    Returns
    -------
    list[dict]
        List of local explanation dicts.
    """
    if indices is None:
        indices = [0, 1, 2, 3, 4]

    results = []
    for idx in indices:
        r = local_explanation(model_path, data_path, idx, output_dir=output_dir)
        results.append(r)
    return results
