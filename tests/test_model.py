"""Tests for model training and loading."""

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.config import RANDOM_SEED
from src.models import (
    evaluate,
    load_model,
    save_model,
    train_baseline,
    train_xgboost,
)


def _make_synthetic_xy(n: int = 300, seed: int = RANDOM_SEED):
    """Create a small synthetic classification dataset."""
    rng = np.random.RandomState(seed)
    X = pd.DataFrame({
        "feat_a": rng.randn(n),
        "feat_b": rng.randn(n),
        "feat_c": rng.uniform(0, 1, n),
        "feat_d": rng.randint(0, 5, n).astype(float),
    })
    y = pd.Series((X["feat_a"] + X["feat_b"] + rng.randn(n) * 0.5 > 0.5).astype(int))
    return X, y


# ── Smoke tests for training ────────────────────────────────────────────

class TestTrainBaseline:
    def test_trains_and_returns_metrics(self):
        X, y = _make_synthetic_xy()
        model, metrics = train_baseline(X, y)
        assert "auc_roc" in metrics
        assert "f1" in metrics
        assert 0.0 <= metrics["auc_roc"] <= 1.0

    def test_predict_returns_correct_shape(self):
        X, y = _make_synthetic_xy()
        model, _ = train_baseline(X, y)
        preds = model.predict(X)
        assert len(preds) == len(X)


class TestTrainXGBoost:
    def test_trains_and_returns_metrics(self):
        X, y = _make_synthetic_xy()
        X_train, X_val = X[:200], X[200:]
        y_train, y_val = y[:200], y[200:]
        model, metrics = train_xgboost(
            X_train, y_train, X_val, y_val,
            config={"n_estimators": 10, "verbosity": 0},
        )
        assert "auc_roc" in metrics
        assert 0.0 <= metrics["auc_roc"] <= 1.0

    def test_predict_proba_shape(self):
        X, y = _make_synthetic_xy()
        model, _ = train_xgboost(
            X[:200], y[:200], X[200:], y[200:],
            config={"n_estimators": 10, "verbosity": 0},
        )
        proba = model.predict_proba(X[200:])
        assert proba.shape == (100, 2)
        assert np.allclose(proba.sum(axis=1), 1.0)


# ── Save / load tests ───────────────────────────────────────────────────

class TestSaveLoadModel:
    def test_save_and_load_roundtrip(self):
        X, y = _make_synthetic_xy()
        model, metrics = train_baseline(X, y)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = save_model(
                model, metrics,
                name="test_model",
                output_dir=Path(tmpdir),
                X_train_rows=len(X),
            )
            assert path.exists()

            # Metadata file should exist
            meta_path = Path(tmpdir) / "test_model_metadata.json"
            assert meta_path.exists()
            with open(meta_path) as f:
                meta = json.load(f)
            assert meta["metrics"]["auc_roc"] > 0
            assert meta["training_rows"] == len(X)

            # Reload and predict
            loaded = load_model(path)
            preds = loaded.predict(X)
            assert len(preds) == len(X)

    def test_load_nonexistent_raises(self):
        with pytest.raises(FileNotFoundError):
            load_model("nonexistent_model.joblib")


# ── Test loading the real model artifact (if present) ────────────────────

class TestRealModelArtifact:
    @pytest.fixture
    def model_path(self):
        p = Path("models/xgb_v1.joblib")
        if not p.exists():
            pytest.skip("Model artifact not found (run training first)")
        return p

    def test_loads_successfully(self, model_path):
        model = load_model(model_path)
        assert hasattr(model, "predict")
        assert hasattr(model, "predict_proba")

    def test_metadata_exists(self, model_path):
        meta_path = model_path.with_name("xgb_v1_metadata.json")
        assert meta_path.exists()
        with open(meta_path) as f:
            meta = json.load(f)
        assert "metrics" in meta
        assert "auc_roc" in meta["metrics"]
        print(f"Test AUC from metadata: {meta['metrics']['auc_roc']}")
