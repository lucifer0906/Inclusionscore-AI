"""Scoring API (FastAPI route handlers).

Accepts raw applicant data, applies the same cleaning and feature
engineering pipeline used during training, runs the model, and returns
a score with an explainable decision.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import shap
from pydantic import BaseModel, Field

from src.config import (
    API_TOKEN, FAIRNESS_ENABLED, MODEL_DIR, MODEL_NAME, SCORE_THRESHOLDS,
)
from src.fairness import AGE_BINS, load_fair_thresholds
from src.features import clean_df, create_features
from src.models import load_model, load_model_metadata

# Pre-computed fallbacks from training data.
# Overridden at runtime if metadata contains preprocessing values.
_DEBT_RATIO_CAP = 4979.08
_INCOME_FILL = 5400.0
_DEP_FILL = 0.0

# Lazy-loaded model and SHAP explainer (initialised on first request)
_model = None
_explainer = None
_fair_thresholds: dict[str, float] = {}


def _get_model():
    """Return the cached model, loading it on first call."""
    global _model, _explainer, _fair_thresholds
    global _DEBT_RATIO_CAP, _INCOME_FILL, _DEP_FILL
    if _model is None:
        model_path = MODEL_DIR / f"{MODEL_NAME}.joblib"
        _model = load_model(model_path)
        _explainer = shap.TreeExplainer(_model)
        if FAIRNESS_ENABLED:
            _fair_thresholds = load_fair_thresholds()
        # Load preprocessing params from metadata if available
        meta = load_model_metadata(model_path)
        preproc = meta.get("preprocessing", {})
        if "debt_ratio_cap" in preproc and preproc["debt_ratio_cap"] is not None:
            _DEBT_RATIO_CAP = preproc["debt_ratio_cap"]
        if "income_fill" in preproc and preproc["income_fill"] is not None:
            _INCOME_FILL = preproc["income_fill"]
        if "dep_fill" in preproc and preproc["dep_fill"] is not None:
            _DEP_FILL = preproc["dep_fill"]
    return _model


def _get_explainer():
    """Return the cached SHAP TreeExplainer."""
    if _explainer is None:
        _get_model()  # triggers lazy init of both
    return _explainer


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
    contribution_unit: str = "log-odds (positive = increases default risk)"


# ── Core scoring logic ───────────────────────────────────────────────────

def _age_group(age: int) -> str:
    """Map age to its fairness group label."""
    for lo, hi, lbl in AGE_BINS:
        if lo <= age <= hi:
            return lbl
    return "unknown"


def _make_decision(prob: float, age: int | None = None) -> str:
    """Map a default probability to APPROVE / REVIEW / REJECT.

    When fairness-aware thresholds are loaded and an age is provided,
    uses the group-specific approval threshold instead of the global one.
    """
    approve_thresh = SCORE_THRESHOLDS["APPROVE"]
    if FAIRNESS_ENABLED and _fair_thresholds and age is not None:
        group = _age_group(age)
        approve_thresh = _fair_thresholds.get(group, approve_thresh)

    if prob <= approve_thresh:
        return "APPROVE"
    elif prob <= SCORE_THRESHOLDS["REVIEW"]:
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

    cleaned = clean_df(raw, fit=False, debt_ratio_cap=_DEBT_RATIO_CAP,
                       income_fill=_INCOME_FILL, dep_fill=_DEP_FILL)
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
    model = _get_model()
    explainer = _get_explainer()

    proba = float(model.predict_proba(X)[0, 1])
    decision = _make_decision(proba, age=applicant.age)

    # Per-applicant SHAP values for genuine explainability
    shap_values = explainer.shap_values(X)
    contributions = shap_values[0]  # single-row array
    feature_names = list(X.columns)
    values = X.iloc[0].values

    contribs = sorted(
        zip(feature_names, values, contributions),
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
