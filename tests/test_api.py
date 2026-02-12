"""Tests for the scoring API."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from api.main import app
from src.config import API_TOKEN


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
    assert resp.status_code == 403  # missing credentials


def test_score_rejects_bad_token(client):
    resp = client.post(
        "/score",
        json=VALID_PAYLOAD,
        headers={"Authorization": "Bearer wrongtoken"},
    )
    assert resp.status_code == 401


# ── Scoring ───────────────────────────────────────────────────────────────


def test_score_valid_payload(client):
    resp = client.post("/score", json=VALID_PAYLOAD, headers=AUTH_HEADER)
    assert resp.status_code == 200
    body = resp.json()
    assert "score" in body
    assert "decision" in body
    assert body["decision"] in ("APPROVE", "REVIEW", "REJECT")
    assert 0.0 <= body["score"] <= 1.0
    assert len(body["top_features"]) == 5


def test_score_response_fields(client):
    resp = client.post("/score", json=VALID_PAYLOAD, headers=AUTH_HEADER)
    body = resp.json()
    assert body["model_version"] == "xgb_v1"
    for feat in body["top_features"]:
        assert "feature" in feat
        assert "value" in feat
        assert "contribution" in feat


def test_score_with_nulls(client):
    """MonthlyIncome and NumberOfDependents accept null."""
    payload = {**VALID_PAYLOAD, "MonthlyIncome": None, "NumberOfDependents": None}
    resp = client.post("/score", json=payload, headers=AUTH_HEADER)
    assert resp.status_code == 200
    assert resp.json()["decision"] in ("APPROVE", "REVIEW", "REJECT")


def test_score_rejects_negative_age(client):
    payload = {**VALID_PAYLOAD, "age": -5}
    resp = client.post("/score", json=payload, headers=AUTH_HEADER)
    assert resp.status_code == 422  # Pydantic validation error


def test_score_rejects_missing_field(client):
    payload = {k: v for k, v in VALID_PAYLOAD.items() if k != "age"}
    resp = client.post("/score", json=payload, headers=AUTH_HEADER)
    assert resp.status_code == 422


# ── Batch scoring ────────────────────────────────────────────────────────


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
    assert resp.status_code == 403


def test_batch_score_empty(client):
    batch_payload = {"applicants": []}
    resp = client.post("/score/batch", json=batch_payload, headers=AUTH_HEADER)
    assert resp.status_code == 200
    assert resp.json()["count"] == 0
