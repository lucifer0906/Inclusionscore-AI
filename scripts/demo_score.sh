#!/usr/bin/env bash
# Demo: score one applicant via the running API.
# Usage:
#   1. Start the server:  uvicorn api.main:app --reload
#   2. Run this script:   bash scripts/demo_score.sh

set -euo pipefail

BASE_URL="${API_URL:-http://localhost:8000}"
TOKEN="${API_TOKEN:-testtoken}"

echo "=== Health check ==="
curl -s "${BASE_URL}/health" | python -m json.tool

echo ""
echo "=== Scoring applicant ==="
curl -s -X POST "${BASE_URL}/score" \
  -H "Authorization: Bearer ${TOKEN}" \
  -H "Content-Type: application/json" \
  -d @scripts/demo_request.json | python -m json.tool
