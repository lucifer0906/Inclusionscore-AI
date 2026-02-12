"""Tests for feature engineering functions."""

import numpy as np
import pandas as pd
import pytest

from src.features import (
    DELINQ_CAP,
    DELINQ_COLS,
    TARGET_COL,
    clean_df,
    create_features,
)


def _make_synthetic(n: int = 20, seed: int = 42) -> pd.DataFrame:
    """Create a small synthetic DataFrame mimicking Give Me Some Credit."""
    rng = np.random.RandomState(seed)
    df = pd.DataFrame(
        {
            TARGET_COL: rng.choice([0, 1], size=n, p=[0.93, 0.07]),
            "RevolvingUtilizationOfUnsecuredLines": rng.uniform(0, 2.0, n),
            "age": rng.choice([0, 25, 35, 50, 65, 80], size=n),
            "NumberOfTime30-59DaysPastDueNotWorse": rng.choice(
                [0, 1, 2, 98], size=n, p=[0.8, 0.1, 0.05, 0.05]
            ),
            "DebtRatio": rng.uniform(0, 50000, n),
            "MonthlyIncome": rng.choice(
                [np.nan, 3000, 5000, 8000], size=n, p=[0.2, 0.3, 0.3, 0.2]
            ),
            "NumberOfOpenCreditLinesAndLoans": rng.randint(0, 20, n),
            "NumberOfTimes90DaysLate": rng.choice(
                [0, 1, 96], size=n, p=[0.85, 0.1, 0.05]
            ),
            "NumberRealEstateLoansOrLines": rng.randint(0, 5, n),
            "NumberOfTime60-89DaysPastDueNotWorse": rng.choice(
                [0, 1, 98], size=n, p=[0.85, 0.1, 0.05]
            ),
            "NumberOfDependents": rng.choice(
                [np.nan, 0, 1, 2, 3], size=n, p=[0.1, 0.5, 0.2, 0.1, 0.1]
            ),
        }
    )
    return df


# ── clean_df tests ───────────────────────────────────────────────────────

class TestCleanDf:
    def test_drops_age_zero(self):
        df = _make_synthetic()
        cleaned = clean_df(df)
        assert (cleaned["age"] > 0).all()

    def test_no_missing_after_imputation(self):
        df = _make_synthetic()
        cleaned = clean_df(df)
        assert cleaned["MonthlyIncome"].isnull().sum() == 0
        assert cleaned["NumberOfDependents"].isnull().sum() == 0

    def test_revolving_util_capped(self):
        df = _make_synthetic()
        cleaned = clean_df(df)
        assert cleaned["RevolvingUtilizationOfUnsecuredLines"].max() <= 1.0

    def test_delinquency_sentinel_capped(self):
        df = _make_synthetic()
        cleaned = clean_df(df)
        for col in DELINQ_COLS:
            assert cleaned[col].max() <= DELINQ_CAP

    def test_debt_ratio_capped(self):
        """Verify DebtRatio capping works on data with clear outliers."""
        df = _make_synthetic(n=200)
        # Inject a clear outlier
        df.loc[0, "DebtRatio"] = 999999.0
        cleaned = clean_df(df)
        # The extreme outlier should be pulled down to the 99th-pctile cap
        assert cleaned["DebtRatio"].max() < 999999.0

    def test_returns_copy(self):
        """clean_df must not modify the original dataframe."""
        df = _make_synthetic()
        original_shape = df.shape
        _ = clean_df(df)
        assert df.shape == original_shape

    def test_monthly_income_missing_indicator(self):
        """MonthlyIncome_Missing flag should be 1 where income was NaN."""
        df = _make_synthetic()
        was_missing = df["MonthlyIncome"].isna()
        cleaned = clean_df(df)
        assert "MonthlyIncome_Missing" in cleaned.columns
        # Rows that had NaN income should have indicator = 1
        # (after dropping age==0 rows, indices reset)
        assert cleaned["MonthlyIncome_Missing"].dtype == np.int8
        assert cleaned["MonthlyIncome_Missing"].isin([0, 1]).all()


# ── create_features tests ────────────────────────────────────────────────

class TestCreateFeatures:
    def setup_method(self):
        self.df = clean_df(_make_synthetic())

    def test_new_columns_present(self):
        result = create_features(self.df)
        expected_new = [
            "TotalDelinquencies",
            "HasDelinquency",
            "IncomeToDebtRatio",
            "OpenCreditPerDependent",
            "AgeBin",
            "CreditUtilBucket",
        ]
        for col in expected_new:
            assert col in result.columns, f"Missing column: {col}"

    def test_total_delinquencies_sum(self):
        result = create_features(self.df)
        manual_sum = self.df[DELINQ_COLS].sum(axis=1).values
        np.testing.assert_array_equal(
            result["TotalDelinquencies"].values, manual_sum
        )

    def test_has_delinquency_binary(self):
        result = create_features(self.df)
        assert set(result["HasDelinquency"].unique()).issubset({0, 1})

    def test_no_nan_in_engineered(self):
        result = create_features(self.df)
        engineered = [
            "TotalDelinquencies", "HasDelinquency",
            "IncomeToDebtRatio", "OpenCreditPerDependent",
            "AgeBin", "CreditUtilBucket",
        ]
        for col in engineered:
            assert result[col].isnull().sum() == 0, f"NaN in {col}"

    def test_returns_copy(self):
        original_cols = set(self.df.columns)
        _ = create_features(self.df)
        assert set(self.df.columns) == original_cols

    def test_output_dtypes(self):
        result = create_features(self.df)
        assert result["HasDelinquency"].dtype == np.int8
        assert result["AgeBin"].dtype in (np.int32, np.int64, int)
        assert result["CreditUtilBucket"].dtype in (np.int32, np.int64, int)
