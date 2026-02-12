"""
Central configuration for InclusionScore AI.

All project-wide constants, paths, and hyperparameters live here so that
every module imports a single source of truth.
"""

import os
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "Datasets"          # raw datasets (user-populated)
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
NOTEBOOK_SAMPLE = PROJECT_ROOT / "notebooks" / "00_quick_start.ipynb"

# ── Reproducibility ─────────────────────────────────────────────────────
RANDOM_SEED = 42

# ── Primary dataset (Give Me Some Credit) ────────────────────────────────
PRIMARY_DATASET_SUBDIR = "give_me_some_credits"
PRIMARY_TRAIN_FILE = "cs-training.csv"
PRIMARY_TEST_FILE = "cs-test.csv"

# ── Model naming ─────────────────────────────────────────────────────────
MODEL_NAME = "xgb_v1"

# ── API / Security ───────────────────────────────────────────────────────
API_TOKEN = os.environ.get("API_TOKEN", "changeme")
SCORE_THRESHOLDS = {
    "APPROVE": 0.3,   # predicted default prob <= 0.3 → approve
    "REVIEW": 0.6,    # 0.3 < prob <= 0.6 → manual review
    # prob > 0.6 → reject
}

# ── Fairness ─────────────────────────────────────────────────────────────
FAIRNESS_ENABLED = os.environ.get("FAIRNESS_ENABLED", "true").lower() == "true"
