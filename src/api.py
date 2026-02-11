"""Scoring API (FastAPI route handlers).

Accepts raw applicant data, applies the same cleaning and feature
engineering pipeline used during training, runs the model, and returns
a score with an explainable decision.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from src.config import API_TOKEN, MODEL_DIR, MODEL_NAME, SCORE_THRESHOLDS
from src.features import clean_df, create_features
from src.models import load_model

# Pre-computed from training data (99th percentile of DebtRatio after
# dropping age==0).  Used by clean_df(fit=False) at inference time.
_DEBT_RATIO_CAP = 4979.08

# Load model once at import time
_model = load_model(MODEL_DIR / f"{MODEL_NAME}.joblib")


# ── Pydantic schemas ─────────────────────────────────────────────────────

class ApplicantInput(BaseModel):
    """Raw applicant fields matching the Give Me Some Credit dataset."""

    RevolvingUtilizationOfUnsecuredLines: float = Field(
        ..., ge=0, description="Total revolving balance / credit limit"
    )
    age: int = Field(..., ge=1, description="Borrower age in years")
    NumberOfTime30_59DaysPastDueNotWorse: int = Field(
        ..., ge=0, alias="NumberOfTime30-59DaysPastDueNotWorse",
        description="Times 30-59 days past due"
    )
    DebtRatio: float = Field(
        ..., ge=0, description="Monthly debt payments / gross income"
    )
    MonthlyIncome: Optional[float] = Field(
        None, ge=0, description="Monthly income (null allowed)"
    )
    NumberOfOpenCreditLinesAndLoans: int = Field(
        ..., ge=0, description="Open loans + credit lines"
    )
    NumberOfTimes90DaysLate: int = Field(
        ..., ge=0, description="Times 90+ days late"
    )
    NumberRealEstateLoansOrLines: int = Field(
        ..., ge=0, description="Mortgage / RE loan count"
    )
    NumberOfTime60_89DaysPastDueNotWorse: int = Field(
        ..., ge=0, alias="NumberOfTime60-89DaysPastDueNotWorse",
        description="Times 60-89 days past due"
    )
    NumberOfDependents: Optional[float] = Field(
        None, ge=0, description="Number of dependents (null allowed)"
    )

    model_config = {"populate_by_name": True}


class FeatureContribution(BaseModel):
    """Single feature contribution in the explanation."""

    feature: str
    value: float
    contribution: float


class ScoreResponse(BaseModel):
    """JSON response from the /score endpoint."""

    model_version: str
    score: float
    decision: str
    top_features: list[FeatureContribution]


# ── Core scoring logic ───────────────────────────────────────────────────

def _make_decision(score: float) -> str:
    """Map a default probability to APPROVE / REVIEW / REJECT."""
    if score <= SCORE_THRESHOLDS["APPROVE"]:
        return "APPROVE"
    elif score <= SCORE_THRESHOLDS["REVIEW"]:
        return "REVIEW"
    return "REJECT"


def prepare_input(applicant: ApplicantInput) -> pd.DataFrame:
    """Convert validated applicant JSON into a model-ready feature row.

    Applies the same ``clean_df`` + ``create_features`` pipeline used
    during training.
    """
    raw = pd.DataFrame([{
        "SeriousDlqin2yrs": 0,  # placeholder target (dropped later)
        "RevolvingUtilizationOfUnsecuredLines": applicant.RevolvingUtilizationOfUnsecuredLines,
        "age": applicant.age,
        "NumberOfTime30-59DaysPastDueNotWorse": applicant.NumberOfTime30_59DaysPastDueNotWorse,
        "DebtRatio": applicant.DebtRatio,
        "MonthlyIncome": applicant.MonthlyIncome,
        "NumberOfOpenCreditLinesAndLoans": applicant.NumberOfOpenCreditLinesAndLoans,
        "NumberOfTimes90DaysLate": applicant.NumberOfTimes90DaysLate,
        "NumberRealEstateLoansOrLines": applicant.NumberRealEstateLoansOrLines,
        "NumberOfTime60-89DaysPastDueNotWorse": applicant.NumberOfTime60_89DaysPastDueNotWorse,
        "NumberOfDependents": applicant.NumberOfDependents,
    }])

    cleaned = clean_df(raw, fit=False, debt_ratio_cap=_DEBT_RATIO_CAP)
    featured = create_features(cleaned)
    featured = featured.drop(columns=["SeriousDlqin2yrs"], errors="ignore")

    # Ensure all columns are numeric (nullable Optional fields may produce
    # object dtype when the input value is None).
    for col in featured.columns:
        featured[col] = pd.to_numeric(featured[col])

    return featured


def score(applicant: ApplicantInput) -> ScoreResponse:
    """Score a single applicant and return structured response."""
    X = prepare_input(applicant)

    proba = float(_model.predict_proba(X)[0, 1])
    decision = _make_decision(proba)

    # Feature contributions via model's internal feature importances
    # (lightweight alternative to running SHAP per request)
    importances = _model.feature_importances_
    feature_names = list(X.columns)
    values = X.iloc[0].values

    contribs = sorted(
        zip(feature_names, values, importances),
        key=lambda t: abs(t[2]),
        reverse=True,
    )

    top_features = [
        FeatureContribution(feature=f, value=float(v), contribution=float(c))
        for f, v, c in contribs[:5]
    ]

    return ScoreResponse(
        model_version=MODEL_NAME,
        score=round(proba, 6),
        decision=decision,
        top_features=top_features,
    )


def verify_token(token: str) -> bool:
    """Check bearer token against configured secret."""
    return token == API_TOKEN
