"""Tests for fairness auditing and threshold recalibration."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.fairness import (
    AGE_BINS,
    DI_THRESHOLD,
    _bin_age,
    calibrate_fair_thresholds,
    compute_demographic_parity,
    compute_disparate_impact,
    compute_subgroup_metrics,
    load_fair_thresholds,
    save_fair_thresholds,
)


def _make_synthetic_fairness_data(n: int = 2000, seed: int = 42):
    """Create synthetic scored data with age groups and probabilities."""
    rng = np.random.RandomState(seed)
    ages = rng.choice([22, 25, 35, 40, 50, 55, 65, 70], size=n)
    # Younger people get systematically higher (worse) probabilities
    proba = np.clip(0.3 + (50 - ages) / 100 + rng.randn(n) * 0.15, 0, 1)
    y_true = (rng.rand(n) < proba * 0.3).astype(int)
    groups = _bin_age(pd.Series(ages))
    return y_true, proba, groups, ages


# ── Age binning ──────────────────────────────────────────────────────────


class TestBinAge:
    def test_known_ages(self):
        ages = pd.Series([20, 35, 50, 65, 5, 130])
        labels = _bin_age(ages)
        assert labels.iloc[0] == "18-30"
        assert labels.iloc[1] == "31-45"
        assert labels.iloc[2] == "46-60"
        assert labels.iloc[3] == "61+"
        assert labels.iloc[4] == "unknown"
        assert labels.iloc[5] == "unknown"


# ── Subgroup metrics ─────────────────────────────────────────────────────


class TestSubgroupMetrics:
    def test_returns_all_groups(self):
        y_true, proba, groups, _ = _make_synthetic_fairness_data()
        df = compute_subgroup_metrics(
            pd.Series(y_true), proba, groups,
        )
        assert set(df["group"]) == {"18-30", "31-45", "46-60", "61+"}

    def test_approval_rates_in_range(self):
        y_true, proba, groups, _ = _make_synthetic_fairness_data()
        df = compute_subgroup_metrics(
            pd.Series(y_true), proba, groups,
        )
        assert all(0 <= r <= 1 for r in df["approval_rate"])


# ── Disparate impact ─────────────────────────────────────────────────────


class TestDisparateImpact:
    def test_di_structure(self):
        y_true, proba, groups, _ = _make_synthetic_fairness_data()
        df = compute_subgroup_metrics(pd.Series(y_true), proba, groups)
        di = compute_disparate_impact(df)
        assert "di_ratio" in di
        assert "pass" in di
        assert 0 <= di["di_ratio"] <= 1


# ── Demographic parity ───────────────────────────────────────────────────


class TestDemographicParity:
    def test_dp_structure(self):
        y_true, proba, groups, _ = _make_synthetic_fairness_data()
        df = compute_subgroup_metrics(pd.Series(y_true), proba, groups)
        dp = compute_demographic_parity(df)
        assert "parity_gap" in dp
        assert "pass" in dp
        assert dp["parity_gap"] >= 0


# ── Threshold calibration ────────────────────────────────────────────────


class TestCalibrateThresholds:
    def test_produces_threshold_per_group(self):
        _, proba, groups, _ = _make_synthetic_fairness_data()
        thresholds = calibrate_fair_thresholds(proba, groups)
        for _, _, lbl in AGE_BINS:
            assert lbl in thresholds
            assert 0 < thresholds[lbl] <= 1.0

    def test_calibrated_di_passes(self):
        y_true, proba, groups, _ = _make_synthetic_fairness_data()
        thresholds = calibrate_fair_thresholds(proba, groups)

        # Compute approval rates with calibrated thresholds
        rows = []
        for g in sorted(groups.unique()):
            mask = groups == g
            thresh = thresholds[g]
            rate = float((proba[mask] <= thresh).mean())
            rows.append({"group": g, "approval_rate": round(rate, 4)})

        fair_df = pd.DataFrame(rows)
        di = compute_disparate_impact(fair_df)
        assert di["pass"], f"DI ratio {di['di_ratio']} < {DI_THRESHOLD}"

    def test_calibrated_parity_passes(self):
        y_true, proba, groups, _ = _make_synthetic_fairness_data()
        thresholds = calibrate_fair_thresholds(proba, groups)

        rows = []
        for g in sorted(groups.unique()):
            mask = groups == g
            thresh = thresholds[g]
            rate = float((proba[mask] <= thresh).mean())
            rows.append({"group": g, "approval_rate": round(rate, 4)})

        fair_df = pd.DataFrame(rows)
        dp = compute_demographic_parity(fair_df)
        assert dp["pass"], f"Parity gap {dp['parity_gap']} > 0.15"


# ── Save / load thresholds ───────────────────────────────────────────────


class TestSaveLoadThresholds:
    def test_roundtrip(self):
        thresholds = {"18-30": 0.55, "31-45": 0.45, "46-60": 0.35, "61+": 0.3}
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test_thresholds.json"
            save_fair_thresholds(thresholds, output_path=path)
            loaded = load_fair_thresholds(path)
            assert loaded == thresholds

    def test_load_missing_returns_empty(self):
        result = load_fair_thresholds(Path("/nonexistent/path.json"))
        assert result == {}
