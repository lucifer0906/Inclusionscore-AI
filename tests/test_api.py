"""Tests for the scoring API."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from api.main import app
from src.config import API_TOKEN

_MODEL_AVAILABLE = Path("models/xgb_v1.joblib").exists()
_skip_no_model = pytest.mark.skipif(
    not _MODEL_AVAILABLE,
    reason="Model artifact not available (skipped in CI)",
)

_ENRICHED_MODEL_AVAILABLE = Path("models/xgb_enriched.joblib").exists()
_skip_no_enriched_model = pytest.mark.skipif(
    not _ENRICHED_MODEL_AVAILABLE,
    reason="Enriched model artifact not available (skipped in CI)",
)


@pytest.fixture()
def client():
    return TestClient(app)


AUTH_HEADER = {"Authorization": f"Bearer {API_TOKEN}"}

VALID_PAYLOAD = {
    "RevolvingUtilizationOfUnsecuredLines": 0.35,
    "age": 42,
    "NumberOfTime30-59DaysPastDueNotWorse": 1,
    "DebtRatio": 0.25,
    "MonthlyIncome": 5500.0,
    "NumberOfOpenCreditLinesAndLoans": 8,
    "NumberOfTimes90DaysLate": 0,
    "NumberRealEstateLoansOrLines": 1,
    "NumberOfTime60-89DaysPastDueNotWorse": 0,
    "NumberOfDependents": 2,
}


# ── Health endpoint ───────────────────────────────────────────────────────


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


# ── Auth ──────────────────────────────────────────────────────────────────


def test_score_requires_auth(client):
    resp = client.post("/score", json=VALID_PAYLOAD)
    assert resp.status_code == 401  # missing credentials


def test_score_rejects_bad_token(client):
    resp = client.post(
        "/score",
        json=VALID_PAYLOAD,
        headers={"Authorization": "Bearer wrongtoken"},
    )
    assert resp.status_code == 401


# ── Scoring ───────────────────────────────────────────────────────────────


@_skip_no_model
def test_score_valid_payload(client):
    resp = client.post("/score", json=VALID_PAYLOAD, headers=AUTH_HEADER)
    assert resp.status_code == 200
    body = resp.json()
    assert "score" in body
    assert "decision" in body
    assert body["decision"] in ("APPROVE", "REVIEW", "REJECT")
    assert 0.0 <= body["score"] <= 1.0
    assert len(body["top_features"]) == 5


@_skip_no_model
def test_score_response_fields(client):
    resp = client.post("/score", json=VALID_PAYLOAD, headers=AUTH_HEADER)
    body = resp.json()
    assert body["model_version"] == "xgb_v1"
    assert "contribution_unit" in body
    for feat in body["top_features"]:
        assert "feature" in feat
        assert "value" in feat
        assert "contribution" in feat


@_skip_no_model
def test_score_with_nulls(client):
    """MonthlyIncome and NumberOfDependents accept null."""
    payload = {**VALID_PAYLOAD, "MonthlyIncome": None, "NumberOfDependents": None}
    resp = client.post("/score", json=payload, headers=AUTH_HEADER)
    assert resp.status_code == 200
    assert resp.json()["decision"] in ("APPROVE", "REVIEW", "REJECT")


@_skip_no_model
def test_score_rejects_negative_age(client):
    payload = {**VALID_PAYLOAD, "age": -5}
    resp = client.post("/score", json=payload, headers=AUTH_HEADER)
    assert resp.status_code == 422  # Pydantic validation error


@_skip_no_model
def test_score_rejects_missing_field(client):
    payload = {k: v for k, v in VALID_PAYLOAD.items() if k != "age"}
    resp = client.post("/score", json=payload, headers=AUTH_HEADER)
    assert resp.status_code == 422


# ── Batch scoring ────────────────────────────────────────────────────────


@_skip_no_model
def test_batch_score_valid(client):
    batch_payload = {"applicants": [VALID_PAYLOAD, VALID_PAYLOAD]}
    resp = client.post("/score/batch", json=batch_payload, headers=AUTH_HEADER)
    assert resp.status_code == 200
    body = resp.json()
    assert body["count"] == 2
    assert len(body["results"]) == 2
    for result in body["results"]:
        assert result["decision"] in ("APPROVE", "REVIEW", "REJECT")
        assert 0.0 <= result["score"] <= 1.0


def test_batch_score_requires_auth(client):
    batch_payload = {"applicants": [VALID_PAYLOAD]}
    resp = client.post("/score/batch", json=batch_payload)
    assert resp.status_code == 401


@_skip_no_model
def test_batch_score_empty(client):
    batch_payload = {"applicants": []}
    resp = client.post("/score/batch", json=batch_payload, headers=AUTH_HEADER)
    assert resp.status_code == 200
    assert resp.json()["count"] == 0


# ── Enriched scoring ────────────────────────────────────────────────────

ENRICHED_PAYLOAD = {
    "AMT_INCOME_TOTAL": 270000.0,
    "AMT_CREDIT": 1293502.5,
    "AMT_ANNUITY": 35698.5,
    "AMT_GOODS_PRICE": 1129500.0,
    "DAYS_BIRTH": -12005,
    "DAYS_EMPLOYED": -4542,
    "DAYS_REGISTRATION": -12563,
    "DAYS_ID_PUBLISH": -4260,
    "EXT_SOURCE_1": 0.5,
    "EXT_SOURCE_2": 0.6,
    "EXT_SOURCE_3": 0.55,
    "CNT_CHILDREN": 0,
    "CNT_FAM_MEMBERS": 2,
    "REGION_RATING_CLIENT": 2,
    "DAYS_LAST_PHONE_CHANGE": -1134,
    "OBS_30_CNT_SOCIAL_CIRCLE": 2,
    "DEF_30_CNT_SOCIAL_CIRCLE": 0,
    # Alternate data features
    "inst_count": 50,
    "inst_avg_delay": -5.2,
    "inst_max_delay": 10,
    "inst_late_ratio": 0.1,
    "inst_early_ratio": 0.4,
    "inst_avg_payment_ratio": 0.98,
    "inst_underpaid_ratio": 0.05,
    "cc_months": 24,
    "cc_avg_utilization": 0.3,
    "bureau_count": 3,
    "bureau_active_ratio": 0.33,
    "bureau_overdue_ratio": 0.0,
}


def test_enriched_requires_auth(client):
    resp = client.post("/score/enriched", json=ENRICHED_PAYLOAD)
    assert resp.status_code == 401


def test_enriched_rejects_bad_token(client):
    resp = client.post(
        "/score/enriched",
        json=ENRICHED_PAYLOAD,
        headers={"Authorization": "Bearer wrongtoken"},
    )
    assert resp.status_code == 401


@_skip_no_enriched_model
def test_enriched_valid_payload(client):
    resp = client.post("/score/enriched", json=ENRICHED_PAYLOAD, headers=AUTH_HEADER)
    assert resp.status_code == 200
    body = resp.json()
    assert "score" in body
    assert "decision" in body
    assert body["decision"] in ("APPROVE", "REVIEW", "REJECT")
    assert 0.0 <= body["score"] <= 1.0
    assert body["model_version"] == "xgb_enriched"
    assert len(body["top_features"]) == 5


@_skip_no_enriched_model
def test_enriched_without_alt_data(client):
    """Scoring with only application features (no alternate data) should still work."""
    app_only_payload = {
        "AMT_INCOME_TOTAL": 270000.0,
        "AMT_CREDIT": 1293502.5,
        "AMT_ANNUITY": 35698.5,
        "AMT_GOODS_PRICE": 1129500.0,
        "DAYS_BIRTH": -12005,
        "DAYS_EMPLOYED": -4542,
        "DAYS_REGISTRATION": -12563,
        "DAYS_ID_PUBLISH": -4260,
        "EXT_SOURCE_1": 0.5,
        "EXT_SOURCE_2": 0.6,
        "EXT_SOURCE_3": 0.55,
        "CNT_CHILDREN": 0,
        "CNT_FAM_MEMBERS": 2,
        "REGION_RATING_CLIENT": 2,
        "DAYS_LAST_PHONE_CHANGE": -1134,
        "OBS_30_CNT_SOCIAL_CIRCLE": 2,
        "DEF_30_CNT_SOCIAL_CIRCLE": 0,
    }
    resp = client.post("/score/enriched", json=app_only_payload, headers=AUTH_HEADER)
    assert resp.status_code == 200
    body = resp.json()
    assert body["decision"] in ("APPROVE", "REVIEW", "REJECT")


@_skip_no_enriched_model
def test_enriched_different_applicants_different_shap(client):
    """Two different applicants should get different SHAP contributions."""
    payload2 = {
        **ENRICHED_PAYLOAD,
        "AMT_INCOME_TOTAL": 50000.0,
        "AMT_CREDIT": 200000.0,
        "inst_late_ratio": 0.8,
        "bureau_overdue_ratio": 0.5,
    }
    resp1 = client.post("/score/enriched", json=ENRICHED_PAYLOAD, headers=AUTH_HEADER)
    resp2 = client.post("/score/enriched", json=payload2, headers=AUTH_HEADER)
    assert resp1.status_code == 200
    assert resp2.status_code == 200
    # SHAP contributions should differ for different inputs
    contribs1 = [f["contribution"] for f in resp1.json()["top_features"]]
    contribs2 = [f["contribution"] for f in resp2.json()["top_features"]]
    assert contribs1 != contribs2
