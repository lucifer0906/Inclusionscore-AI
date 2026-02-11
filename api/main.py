"""FastAPI application entrypoint.

Exposes:
- GET  /health  -- liveness check (no auth)
- POST /score   -- score a single applicant (bearer-token auth)
"""

from __future__ import annotations

from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from src.api import ApplicantInput, ScoreResponse, score, verify_token

app = FastAPI(
    title="InclusionScore AI",
    description="AI-powered alternate credit scoring for the unbanked / under-banked.",
    version="0.1.0",
)

_bearer_scheme = HTTPBearer()


def _authenticate(
    credentials: HTTPAuthorizationCredentials = Depends(_bearer_scheme),
) -> str:
    """Validate the bearer token and return it."""
    if not verify_token(credentials.credentials):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API token.",
        )
    return credentials.credentials


# ── Routes ────────────────────────────────────────────────────────────────


@app.get("/health")
def health() -> dict:
    """Liveness / readiness probe."""
    return {"status": "ok"}


@app.post("/score", response_model=ScoreResponse)
def score_applicant(
    applicant: ApplicantInput,
    _token: str = Depends(_authenticate),
) -> ScoreResponse:
    """Score a single applicant and return the decision + explanation."""
    return score(applicant)
