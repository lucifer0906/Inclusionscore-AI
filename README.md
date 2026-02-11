# InclusionScore AI

AI-powered alternate credit scoring for unbanked / under-banked individuals.

> **HackerEarth Hack-o-Hire 2026 | Theme 4: AI-Powered Alternate Credit Scoring**

---

## Problem Statement

Traditional credit scoring relies on lengthy credit histories, excluding ~1.4 billion adults worldwide who lack formal banking access. InclusionScore AI builds a transparent, fair, and explainable credit scoring model using alternative financial behaviour signals -- enabling lenders to assess creditworthiness for the unbanked and under-banked.

## Key Features

- **XGBoost classifier** with AUC-ROC ~0.860 on held-out test data
- **No protected attributes** -- model uses only financial behaviour features
- **SHAP explainability** -- global feature importance + per-applicant decision breakdowns
- **Fairness auditing** -- disparate impact and demographic parity analysis across age groups
- **REST API** (FastAPI) for real-time scoring with bearer-token authentication
- **Docker-ready** for production deployment
- **28 automated tests** covering features, model, and API

## Architecture

```
Raw CSV Data
    |
    v
clean_df()           -- impute, cap outliers, remove sentinels
    |
    v
create_features()    -- 6 engineered features
    |
    v
XGBoost Model        -- scale_pos_weight=14, max_depth=5
    |
    v
score + SHAP         -- probability, decision, top feature contributions
    |
    v
FastAPI /score       -- { score, decision, top_features }
```

## Project Structure

```
InclusionScore-AI/
|-- src/
|   |-- config.py           # Central configuration (paths, constants, thresholds)
|   |-- data_loader.py      # Dataset loading and inspection utilities
|   |-- features.py         # Cleaning pipeline + feature engineering
|   |-- models.py           # LR baseline + XGBoost training, evaluation, persistence
|   |-- explainability.py   # SHAP global and local explanations
|   |-- fairness.py         # Disparate impact + demographic parity audit
|   |-- api.py              # Scoring logic, Pydantic schemas, input preparation
|   |-- train.py            # Training orchestration script
|   `-- utils.py            # General utilities
|-- api/
|   `-- main.py             # FastAPI app with /health and /score endpoints
|-- tests/
|   |-- test_features.py    # 12 tests for cleaning + feature engineering
|   |-- test_model.py       # 8 tests for training, save/load, artifact validation
|   `-- test_api.py         # 8 tests for API endpoints, auth, validation
|-- notebooks/
|   |-- 00_quick_start.ipynb # Unified end-to-end pipeline (hackathon submission)
|   |-- 01_EDA.ipynb         # Exploratory data analysis
|   |-- 03_modeling.ipynb    # Model training and evaluation
|   `-- 04_explain_fairness.ipynb  # Explainability and fairness
|-- reports/
|   |-- eda_summary.md      # EDA findings and cleaning plan
|   |-- fairness_report.md  # Fairness audit results
|   |-- privacy_notes.md    # Privacy considerations
|   |-- eda_plots/           # 11 PNG plots (target dist, correlations, etc.)
|   `-- explainability/      # 8 PNG plots (SHAP summary, beeswarm, waterfalls)
|-- models/
|   |-- xgb_v1.joblib       # Trained XGBoost model (gitignored)
|   `-- xgb_v1_metadata.json# Model metadata (metrics, params, row counts)
|-- scripts/
|   |-- demo_score.sh       # Curl-based API demo
|   `-- demo_request.json   # Sample applicant payload
|-- Dockerfile
|-- docker-compose.yml
|-- requirements.txt
`-- .github/workflows/ci.yml
```

## Quick Start

### Prerequisites

- Python 3.10+
- The [Give Me Some Credit](https://www.kaggle.com/c/GiveMeSomeCredit) dataset placed in `Datasets/give_me_some_credits/`

### Installation

```bash
git clone https://github.com/lucifer0906/Inclusionscore-AI.git
cd Inclusionscore-AI

python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/Mac

pip install -r requirements.txt
```

### Train the Model

```bash
python -m src.train
```

This runs the full pipeline: load raw data, clean, engineer features, split (70/15/15), train LR baseline + XGBoost, save model artifact to `models/xgb_v1.joblib`.

### Run Tests

```bash
pytest tests/ -v
```

### Start the API

```bash
uvicorn api.main:app --reload
```

Then test:

```bash
# Health check
curl http://localhost:8000/health

# Score an applicant
curl -X POST http://localhost:8000/score \
  -H "Authorization: Bearer testtoken" \
  -H "Content-Type: application/json" \
  -d @scripts/demo_request.json
```

### Run with Docker

```bash
docker-compose up --build
```

### Run the Unified Notebook

```bash
cd notebooks
jupyter notebook 00_quick_start.ipynb
```

Run all cells top-to-bottom for the complete end-to-end pipeline.

## Dataset

**Give Me Some Credit** (Kaggle) -- 150,000 borrowers with 10 financial features and a binary default target (`SeriousDlqin2yrs`).

| Feature | Description |
|---------|-------------|
| RevolvingUtilizationOfUnsecuredLines | Credit card balance / credit limit |
| age | Borrower age in years |
| NumberOfTime30-59DaysPastDueNotWorse | Count of 30-59 day late payments |
| DebtRatio | Monthly debt payments / monthly income |
| MonthlyIncome | Monthly income (19.8% missing) |
| NumberOfOpenCreditLinesAndLoans | Open loans and credit lines |
| NumberOfTimes90DaysLate | Count of 90+ day late payments |
| NumberRealEstateLoansOrLines | Mortgage and real estate loans |
| NumberOfTime60-89DaysPastDueNotWorse | Count of 60-89 day late payments |
| NumberOfDependents | Number of dependents (2.6% missing) |

**Target:** `SeriousDlqin2yrs` -- 1 if borrower was 90+ days past due within 2 years (6.7% positive rate).

## Feature Engineering

6 derived features are created from the raw data:

| Feature | Logic |
|---------|-------|
| TotalDelinquencies | Sum of all three past-due count columns |
| HasDelinquency | Binary flag: 1 if any past-due count > 0 |
| IncomeToDebtRatio | MonthlyIncome / (DebtRatio + 1) |
| OpenCreditPerDependent | OpenCreditLines / (Dependents + 1) |
| AgeBin | Binned age: [0-30), [30-45), [45-60), [60-120] |
| CreditUtilBucket | Binned revolving utilization: [0-0.25), [0.25-0.50), [0.50-0.75), [0.75-1.0] |

## Model Performance

| Metric | Logistic Regression | XGBoost |
|--------|--------------------:|--------:|
| AUC-ROC | 0.857 | **0.860** |
| Precision | 0.211 | **0.232** |
| Recall | **0.766** | 0.737 |
| F1-Score | 0.331 | **0.352** |
| Brier Score | 0.148 | **0.128** |

## Explainability

SHAP (SHapley Additive exPlanations) provides two levels of transparency:

- **Global:** Which features matter most across all predictions (bar plot + beeswarm)
- **Local:** For each individual applicant, which features pushed their score up or down (waterfall plots)

Top features by global SHAP importance:
1. RevolvingUtilizationOfUnsecuredLines
2. TotalDelinquencies
3. age
4. DebtRatio
5. NumberOfTimes90DaysLate

## Fairness Audit

Using age-based subgroups as proxy (no explicit protected attributes in the dataset):

| Metric | Value | Threshold | Status |
|--------|-------|-----------|--------|
| Disparate Impact (4/5 rule) | 0.44 | >= 0.80 | FAIL |
| Demographic Parity gap | 0.48 | <= 0.15 | FAIL |

Younger borrowers (18-30) have lower approval rates than older borrowers (61+). Documented mitigation strategies include threshold re-calibration per subgroup and post-processing probability adjustments.

## API Reference

### `GET /health`

Liveness probe. No authentication required.

**Response:** `{"status": "ok"}`

### `POST /score`

Score a single applicant. Requires `Authorization: Bearer <token>` header.

**Request body:**
```json
{
  "RevolvingUtilizationOfUnsecuredLines": 0.35,
  "age": 42,
  "NumberOfTime30-59DaysPastDueNotWorse": 1,
  "DebtRatio": 0.25,
  "MonthlyIncome": 5500.0,
  "NumberOfOpenCreditLinesAndLoans": 8,
  "NumberOfTimes90DaysLate": 0,
  "NumberRealEstateLoansOrLines": 1,
  "NumberOfTime60-89DaysPastDueNotWorse": 0,
  "NumberOfDependents": 2
}
```

**Response:**
```json
{
  "model_version": "xgb_v1",
  "score": 0.1234,
  "decision": "APPROVE",
  "top_features": [
    {"feature": "RevolvingUtilizationOfUnsecuredLines", "value": 0.35, "contribution": 0.0821},
    ...
  ]
}
```

**Decision thresholds:**

| Default Probability | Decision |
|---------------------|----------|
| <= 0.30 | APPROVE |
| 0.30 - 0.60 | REVIEW |
| > 0.60 | REJECT |

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.10+ |
| ML Model | XGBoost 2.0 |
| Baseline | scikit-learn (Logistic Regression) |
| Explainability | SHAP 0.43 |
| API | FastAPI + Uvicorn |
| Validation | Pydantic 2.0 |
| Data | pandas, numpy, pyarrow |
| Visualization | matplotlib, seaborn |
| Testing | pytest |
| Containerization | Docker |
| CI | GitHub Actions |

## License

This project was built for the HackerEarth Hack-o-Hire 2026 hackathon.
