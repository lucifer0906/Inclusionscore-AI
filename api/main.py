"""FastAPI application entrypoint.

Exposes:
- GET  /health          -- liveness check (no auth)
- POST /score           -- score a single applicant (bearer-token auth)
- POST /score/batch     -- score multiple applicants (bearer-token auth)
- POST /score/enriched  -- score with enriched model (bearer-token auth)
"""

from __future__ import annotations

from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel

from src.api import ApplicantInput, ScoreResponse, score, verify_token
from src.api_enriched import EnrichedApplicantInput, score_enriched

app = FastAPI(
    title="InclusionScore AI",
    description="AI-powered alternate credit scoring for the unbanked / under-banked.",
    version="0.1.0",
)

_bearer_scheme = HTTPBearer(auto_error=False)


def _authenticate(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer_scheme),
) -> str:
    """Validate the bearer token and return it."""
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing authorization credentials.",
        )
    if not verify_token(credentials.credentials):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API token.",
        )
    return credentials.credentials


# ── Schemas ──────────────────────────────────────────────────────────────


class BatchRequest(BaseModel):
    """Request body for batch scoring."""
    applicants: list[ApplicantInput]


class BatchResponse(BaseModel):
    """Response body for batch scoring."""
    results: list[ScoreResponse]
    count: int


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


@app.post("/score/batch", response_model=BatchResponse)
def score_batch(
    batch: BatchRequest,
    _token: str = Depends(_authenticate),
) -> BatchResponse:
    """Score multiple applicants in a single request."""
    if len(batch.applicants) > 100:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Batch size limited to 100 applicants.",
        )
    results = [score(applicant) for applicant in batch.applicants]
    return BatchResponse(results=results, count=len(results))


@app.post("/score/enriched", response_model=ScoreResponse)
def score_enriched_applicant(
    applicant: EnrichedApplicantInput,
    _token: str = Depends(_authenticate),
) -> ScoreResponse:
    """Score an applicant using the enriched model (application + alternate data)."""
    return score_enriched(applicant)
