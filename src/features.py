"""Feature engineering and cleaning pipeline.

All transformations applied to the raw *Give Me Some Credit* dataset
live here so they can be reused identically at training time and at
inference time (via the API).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.config import PROCESSED_DIR

# ── Constants derived from EDA (Phase 1) ─────────────────────────────────
TARGET_COL = "SeriousDlqin2yrs"

DELINQ_COLS = [
    "NumberOfTime30-59DaysPastDueNotWorse",
    "NumberOfTime60-89DaysPastDueNotWorse",
    "NumberOfTimes90DaysLate",
]

# Cap thresholds chosen from EDA percentiles / domain knowledge
DELINQ_CAP = 15           # sentinel values 96/98 are errors
REVOLVING_UTIL_CAP = 1.0  # utilization above 100 % is anomalous
DEBT_RATIO_CAP_PCTILE = 0.99  # cap at 99th percentile (computed on training data)


# ── Cleaning ─────────────────────────────────────────────────────────────

def clean_df(df: pd.DataFrame, *, fit: bool = True,
             debt_ratio_cap: float | None = None) -> pd.DataFrame:
    """Clean the raw dataframe according to the EDA plan.

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataframe (must contain all original columns).
    fit : bool
        If ``True`` (training), compute the debt-ratio cap from data.
        If ``False`` (inference), *debt_ratio_cap* must be provided.
    debt_ratio_cap : float | None
        Pre-computed cap for ``DebtRatio`` (used when ``fit=False``).

    Returns
    -------
    pd.DataFrame
        Cleaned copy of the input.
    """
    df = df.copy()

    # 1. Drop rows with age == 0 (invalid)
    df = df[df["age"] > 0].reset_index(drop=True)

    # 2. Impute missing values
    df["MonthlyIncome"] = df["MonthlyIncome"].fillna(df["MonthlyIncome"].median())
    df["NumberOfDependents"] = df["NumberOfDependents"].fillna(
        df["NumberOfDependents"].mode()[0]
    )

    # 3. Cap outliers – revolving utilization
    df["RevolvingUtilizationOfUnsecuredLines"] = df[
        "RevolvingUtilizationOfUnsecuredLines"
    ].clip(upper=REVOLVING_UTIL_CAP)

    # 4. Cap outliers – debt ratio
    if fit:
        debt_ratio_cap = float(
            df["DebtRatio"].quantile(DEBT_RATIO_CAP_PCTILE)
        )
    if debt_ratio_cap is None:
        raise ValueError("debt_ratio_cap must be provided when fit=False")
    df["DebtRatio"] = df["DebtRatio"].clip(upper=debt_ratio_cap)

    # 5. Cap delinquency sentinel values (96, 98 -> DELINQ_CAP)
    for col in DELINQ_COLS:
        df[col] = df[col].clip(upper=DELINQ_CAP)

    return df


# ── Feature engineering ──────────────────────────────────────────────────

def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create engineered features from the cleaned dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned dataframe (output of :func:`clean_df`).

    Returns
    -------
    pd.DataFrame
        Dataframe with original + new engineered columns.
    """
    df = df.copy()

    # Total delinquencies (sum of all past-due columns)
    df["TotalDelinquencies"] = df[DELINQ_COLS].sum(axis=1)

    # Binary flag: any delinquency at all
    df["HasDelinquency"] = (df["TotalDelinquencies"] > 0).astype(np.int8)

    # Income-to-debt affordability proxy
    df["IncomeToDebtRatio"] = df["MonthlyIncome"] / (df["DebtRatio"] + 1.0)

    # Open credit lines per household member
    df["OpenCreditPerDependent"] = df["NumberOfOpenCreditLinesAndLoans"] / (
        df["NumberOfDependents"] + 1.0
    )

    # Age bins (captures non-linear age effects)
    df["AgeBin"] = pd.cut(
        df["age"],
        bins=[0, 30, 45, 60, 120],
        labels=[0, 1, 2, 3],
    ).astype(int)

    # Credit utilization buckets
    df["CreditUtilBucket"] = pd.cut(
        df["RevolvingUtilizationOfUnsecuredLines"],
        bins=[-0.01, 0.25, 0.50, 0.75, 1.01],
        labels=[0, 1, 2, 3],
    ).astype(int)

    return df


# ── Persistence ──────────────────────────────────────────────────────────

def save_processed(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_val: pd.Series,
    y_test: pd.Series,
    output_dir: Path | None = None,
) -> None:
    """Save train / val / test splits as Parquet files.

    Saved files::

        output_dir/
            train.parquet   (X_train + y_train)
            val.parquet     (X_val   + y_val)
            test.parquet    (X_test  + y_test)

    Parameters
    ----------
    X_train, X_val, X_test : pd.DataFrame
        Feature matrices.
    y_train, y_val, y_test : pd.Series
        Target vectors.
    output_dir : Path | None
        Destination folder. Defaults to ``PROCESSED_DIR``.
    """
    output_dir = output_dir or PROCESSED_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    for name, X, y in [
        ("train", X_train, y_train),
        ("val", X_val, y_val),
        ("test", X_test, y_test),
    ]:
        combined = X.copy()
        combined[TARGET_COL] = y.values
        combined.to_parquet(output_dir / f"{name}.parquet", index=False)


def load_processed(
    split: str = "train",
    output_dir: Path | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """Load a previously saved processed split.

    Parameters
    ----------
    split : str
        One of ``"train"``, ``"val"``, or ``"test"``.
    output_dir : Path | None
        Folder containing the parquet files.

    Returns
    -------
    tuple[pd.DataFrame, pd.Series]
        ``(X, y)`` pair.
    """
    output_dir = output_dir or PROCESSED_DIR
    df = pd.read_parquet(output_dir / f"{split}.parquet")
    y = df.pop(TARGET_COL)
    return df, y
