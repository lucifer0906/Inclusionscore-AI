"""Scoring API logic for the enriched model (Home Credit + alternate data).

Mirrors the structure of ``src.api`` but serves the ``xgb_enriched`` model
which was trained on application features *plus* alternate data features
(installment payments, credit card usage, POS transactions, previous
applications, and bureau records).

The enriched model uses a different feature set than the primary model,
so it has its own Pydantic schema, input preparation, and scoring function.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import shap
from pydantic import BaseModel, Field

from src.api import FeatureContribution, ScoreResponse
from src.config import MODEL_DIR, SCORE_THRESHOLDS
from src.models import load_model, load_model_metadata

# ── Constants ────────────────────────────────────────────────────────────

_ENRICHED_MODEL_NAME = "xgb_enriched"

# Lazy-loaded model, explainer, and metadata
_model = None
_explainer = None
_feature_columns: list[str] = []


def _get_model():
    """Return the cached enriched model, loading it on first call."""
    global _model, _explainer, _feature_columns
    if _model is None:
        model_path = MODEL_DIR / f"{_ENRICHED_MODEL_NAME}.joblib"
        _model = load_model(model_path)
        _explainer = shap.TreeExplainer(_model)
        meta = load_model_metadata(model_path)
        _feature_columns = meta.get("feature_columns", [])
    return _model


def _get_explainer():
    """Return the cached SHAP TreeExplainer for the enriched model."""
    if _explainer is None:
        _get_model()
    return _explainer


# ── Pydantic schema ─────────────────────────────────────────────────────

class EnrichedApplicantInput(BaseModel):
    """Input schema for scoring with the enriched (Home Credit) model.

    Contains the 17 application-level features plus 32 optional alternate
    data features.  When alternate data fields are omitted (null), they
    default to 0.0 — the same fill value used during training for
    applicants with no alternate data records.
    """

    # -- Application features (required) ----------------------------------
    AMT_INCOME_TOTAL: float = Field(..., description="Total income")
    AMT_CREDIT: float = Field(..., description="Credit amount of the loan")
    AMT_ANNUITY: float = Field(..., description="Loan annuity")
    AMT_GOODS_PRICE: float = Field(..., description="Price of goods for which loan is given")
    DAYS_BIRTH: float = Field(..., description="Client age in days (negative)")
    DAYS_EMPLOYED: float = Field(..., description="Days employed (negative)")
    DAYS_REGISTRATION: float = Field(..., description="Days since registration")
    DAYS_ID_PUBLISH: float = Field(..., description="Days since ID publish")
    EXT_SOURCE_1: Optional[float] = Field(None, description="External source score 1")
    EXT_SOURCE_2: Optional[float] = Field(None, description="External source score 2")
    EXT_SOURCE_3: Optional[float] = Field(None, description="External source score 3")
    CNT_CHILDREN: float = Field(0, description="Number of children")
    CNT_FAM_MEMBERS: float = Field(1, description="Family member count")
    REGION_RATING_CLIENT: float = Field(2, description="Region rating of client")
    DAYS_LAST_PHONE_CHANGE: float = Field(0, description="Days since last phone change")
    OBS_30_CNT_SOCIAL_CIRCLE: Optional[float] = Field(None, description="30-day social circle observations")
    DEF_30_CNT_SOCIAL_CIRCLE: Optional[float] = Field(None, description="30-day social circle defaults")

    # -- Installment payment features (optional) --------------------------
    inst_count: Optional[float] = Field(None, description="Number of installment records")
    inst_avg_delay: Optional[float] = Field(None, description="Average payment delay (days)")
    inst_max_delay: Optional[float] = Field(None, description="Maximum payment delay (days)")
    inst_late_ratio: Optional[float] = Field(None, description="Ratio of late payments")
    inst_early_ratio: Optional[float] = Field(None, description="Ratio of early payments")
    inst_avg_payment_ratio: Optional[float] = Field(None, description="Average payment-to-instalment ratio")
    inst_underpaid_ratio: Optional[float] = Field(None, description="Ratio of underpaid instalments")

    # -- Credit card features (optional) ----------------------------------
    cc_months: Optional[float] = Field(None, description="Months of credit card history")
    cc_avg_utilization: Optional[float] = Field(None, description="Average credit utilization")
    cc_max_utilization: Optional[float] = Field(None, description="Maximum credit utilization")
    cc_avg_balance: Optional[float] = Field(None, description="Average credit card balance")
    cc_avg_drawings: Optional[float] = Field(None, description="Average drawings")
    cc_avg_atm_drawings: Optional[float] = Field(None, description="Average ATM drawings")
    cc_avg_payment: Optional[float] = Field(None, description="Average payment on credit card")
    cc_max_dpd: Optional[float] = Field(None, description="Max days past due on credit card")
    cc_avg_dpd: Optional[float] = Field(None, description="Average days past due on credit card")

    # -- POS/cash loan features (optional) --------------------------------
    pos_months: Optional[float] = Field(None, description="Months of POS loan history")
    pos_completed_ratio: Optional[float] = Field(None, description="Ratio of completed POS loans")
    pos_max_dpd: Optional[float] = Field(None, description="Max days past due on POS loans")
    pos_avg_dpd: Optional[float] = Field(None, description="Average days past due on POS loans")
    pos_avg_instalments_left: Optional[float] = Field(None, description="Average remaining instalments")

    # -- Previous application features (optional) -------------------------
    prev_app_count: Optional[float] = Field(None, description="Number of previous applications")
    prev_approved_ratio: Optional[float] = Field(None, description="Ratio of approved applications")
    prev_refused_ratio: Optional[float] = Field(None, description="Ratio of refused applications")
    prev_avg_credit: Optional[float] = Field(None, description="Average credit of previous apps")
    prev_avg_annuity: Optional[float] = Field(None, description="Average annuity of previous apps")

    # -- Bureau (external credit) features (optional) ---------------------
    bureau_count: Optional[float] = Field(None, description="Number of bureau records")
    bureau_active_ratio: Optional[float] = Field(None, description="Ratio of active bureau credits")
    bureau_overdue_ratio: Optional[float] = Field(None, description="Ratio of overdue bureau records")
    bureau_avg_credit: Optional[float] = Field(None, description="Average bureau credit sum")
    bureau_max_overdue_amt: Optional[float] = Field(None, description="Maximum overdue amount in bureau")
    bureau_avg_debt: Optional[float] = Field(None, description="Average bureau debt")

    model_config = {"populate_by_name": True}


# ── Core scoring logic ──────────────────────────────────────────────────

def _make_enriched_decision(prob: float) -> str:
    """Map a default probability to APPROVE / REVIEW / REJECT."""
    if prob <= SCORE_THRESHOLDS["APPROVE"]:
        return "APPROVE"
    elif prob <= SCORE_THRESHOLDS["REVIEW"]:
        return "REVIEW"
    return "REJECT"


def prepare_enriched_input(applicant: EnrichedApplicantInput) -> pd.DataFrame:
    """Convert validated applicant JSON into an enriched model-ready feature row.

    All optional (alternate data) fields that are None are filled with 0.0,
    matching the training pipeline's ``fillna(0)`` for applicants without
    alternate data records.
    """
    model = _get_model()  # ensures _feature_columns is loaded
    data = applicant.model_dump()

    # Build a single-row DataFrame with columns in the training order
    row = {}
    for col in _feature_columns:
        val = data.get(col)
        row[col] = 0.0 if val is None else float(val)

    df = pd.DataFrame([row])
    return df


def score_enriched(applicant: EnrichedApplicantInput) -> ScoreResponse:
    """Score a single applicant using the enriched model and return structured response."""
    X = prepare_enriched_input(applicant)
    model = _get_model()
    explainer = _get_explainer()

    proba = float(model.predict_proba(X)[0, 1])
    decision = _make_enriched_decision(proba)

    # Per-applicant SHAP values for explainability
    shap_values = explainer.shap_values(X)
    contributions = shap_values[0]
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
        model_version=_ENRICHED_MODEL_NAME,
        score=round(proba, 6),
        decision=decision,
        top_features=top_features,
    )
