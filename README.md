# InclusionScore AI

AI-powered alternate credit scoring for unbanked / under-banked individuals.

> **HackerEarth Hack-o-Hire 2026 | Theme 4: AI-Powered Alternate Credit Scoring**

[![CI](https://github.com/lucifer0906/Inclusionscore-AI/actions/workflows/ci.yml/badge.svg?branch=master)](https://github.com/lucifer0906/Inclusionscore-AI/actions/workflows/ci.yml)

---

## Problem Statement

Traditional credit scoring relies on lengthy credit histories, excluding ~1.4 billion adults worldwide who lack formal banking access. InclusionScore AI builds a transparent, fair, and explainable credit scoring model using alternative financial behaviour signals -- enabling lenders to assess creditworthiness for the unbanked and under-banked.

## Key Features

- **Optuna-tuned XGBoost** with AUC-ROC **0.848** on held-out test data (50-trial Bayesian search, 5-fold CV)
- **SMOTE + scale_pos_weight** dual strategy for class imbalance (6.7% default rate)
- **Per-applicant SHAP explainability** -- real-time TreeExplainer values per scoring request
- **Fairness-aware decisioning** -- age-group threshold calibration achieving DI = 0.856 (PASS) and parity gap = 0.14 (PASS)
- **Counterfactual explanations** -- actionable suggestions for rejected applicants ("increase income by $X to qualify")
- **REST API** (FastAPI) with single + batch scoring and bearer-token authentication
- **Streamlit dashboard** for interactive scoring, SHAP visualization, and counterfactual display
- **Docker-ready** for production deployment
- **Alternate data integration** -- 32 features engineered from 5 Home Credit transaction tables (30M+ records) with demonstrated +0.019 AUC improvement
- **48 automated tests** covering features, model, fairness, and API (incl. batch + enriched)

## Architecture

```
Raw CSV Data
    |
    v
clean_df()           -- impute, cap outliers, MonthlyIncome_Missing indicator
    |
    v
create_features()    -- 7 engineered features
    |
    v
SMOTE (train only)   -- oversample minority class for balanced training
    |
    v
XGBoost Model        -- Optuna-tuned (max_depth=4, lr=0.018, 750 trees)
    |
    v
score + SHAP         -- probability, decision, per-applicant contributions
    |
    v
Fair Thresholds      -- age-group-specific approval thresholds (DI >= 0.80)
    |
    v
FastAPI              -- /score, /score/batch, /score/enriched, /health
    |
    v
Streamlit Dashboard  -- interactive form + SHAP chart + counterfactuals
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
|   |-- fairness.py         # Disparate impact, demographic parity, threshold calibration
|   |-- api.py              # Scoring logic, Pydantic schemas, input preparation
|   |-- api_enriched.py     # Enriched model scoring (Home Credit + alternate data)
|   |-- train.py            # Training orchestration (data -> train -> evaluate -> save)
|   |-- tuning.py           # Optuna hyperparameter optimization (50 trials, 5-fold CV)
|   |-- counterfactual.py   # Counterfactual explanations for denied applicants
|   |-- alternate_data.py   # Alternate data feature engineering (Home Credit)
|   `-- utils.py            # General utilities
|-- api/
|   `-- main.py             # FastAPI app with /health, /score, /score/batch, /score/enriched
|-- app/
|   `-- dashboard.py        # Streamlit interactive dashboard
|-- tests/
|   |-- test_features.py    # 14 tests for cleaning + feature engineering
|   |-- test_model.py       # 8 tests for training, save/load, artifact validation
|   |-- test_fairness.py    # 10 tests for fairness auditing + calibration
|   `-- test_api.py         # 16 tests for API endpoints, auth, batch + enriched scoring
|-- notebooks/
|   |-- 00_quick_start.ipynb # Unified end-to-end pipeline (hackathon submission)
|   |-- 01_EDA.ipynb         # Exploratory data analysis
|   |-- 02_modeling.ipynb    # Model training and evaluation
|   `-- 03_explain_fairness.ipynb  # Explainability and fairness
|-- reports/
|   |-- eda_summary.md      # EDA findings and cleaning plan
|   |-- fairness_report.md  # Fairness audit results (pre + post mitigation)
|   |-- privacy_notes.md    # Privacy considerations
|   |-- eda_plots/           # 11 PNG plots (target dist, correlations, etc.)
|   `-- explainability/      # SHAP plots (summary, beeswarm, waterfalls, subgroup AUC)
|-- models/
|   |-- xgb_v1.joblib       # Trained XGBoost model (gitignored)
|   |-- xgb_v1_metadata.json# Model metadata (metrics, params, preprocessing, threshold)
|   |-- xgb_enriched.joblib # Enriched model with alternate data (gitignored)
|   |-- xgb_enriched_metadata.json # Enriched model metadata
|   |-- best_params.json    # Optuna-tuned hyperparameters
|   `-- fair_thresholds.json# Per-age-group calibrated approval thresholds
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

This runs the full pipeline: load raw data, clean, engineer features, apply SMOTE to training set, split (70/15/15), train LR baseline + Optuna-tuned XGBoost, compute optimal threshold, and save model artifact to `models/xgb_v1.joblib`.

### Hyperparameter Tuning (Optional)

```bash
python -c "from src.train import prepare_data; from src.tuning import tune_xgboost; import pandas as pd; X_train, X_val, X_test, y_train, y_val, y_test = prepare_data(); X = pd.concat([X_train, X_val]).values; y = pd.concat([y_train, y_val]).values; tune_xgboost(X, y, n_trials=50)"
```

Results are saved to `models/best_params.json` and automatically used by the training pipeline.

### Run Tests

```bash
pytest tests/ -v
```

48 tests covering feature engineering, model training, fairness calibration, and API endpoints (incl. batch + enriched scoring).

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
  -H "Authorization: Bearer changeme" \
  -H "Content-Type: application/json" \
  -d @scripts/demo_request.json

# Batch scoring (up to 100 applicants)
curl -X POST http://localhost:8000/score/batch \
  -H "Authorization: Bearer changeme" \
  -H "Content-Type: application/json" \
  -d '{"applicants": [{"RevolvingUtilizationOfUnsecuredLines":0.35,"age":42,"NumberOfTime30-59DaysPastDueNotWorse":1,"DebtRatio":0.25,"MonthlyIncome":5500,"NumberOfOpenCreditLinesAndLoans":8,"NumberOfTimes90DaysLate":0,"NumberRealEstateLoansOrLines":1,"NumberOfTime60-89DaysPastDueNotWorse":0,"NumberOfDependents":2}]}'
```

### Launch the Streamlit Dashboard

```bash
streamlit run app/dashboard.py
```

Interactive form for applicant data entry with real-time scoring, SHAP waterfall visualization, and counterfactual suggestions for denied applicants.

### Run with Docker

```bash
docker-compose up --build
```

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

7 derived features are created from the raw data:

| Feature | Logic |
|---------|-------|
| MonthlyIncome_Missing | Binary indicator: 1 if original MonthlyIncome was null (before imputation) |
| TotalDelinquencies | Sum of all three past-due count columns |
| HasDelinquency | Binary flag: 1 if any past-due count > 0 |
| IncomeToDebtRatio | MonthlyIncome / (DebtRatio + 1) |
| OpenCreditPerDependent | OpenCreditLines / (Dependents + 1) |
| AgeBin | Binned age: [0-30), [30-45), [45-60), [60-120] |
| CreditUtilBucket | Binned revolving utilization: [0-0.25), [0.25-0.50), [0.50-0.75), [0.75-1.0] |

## Class Imbalance Handling

The target has a 6.7% positive rate (~1:14 imbalance). We use a **dual strategy**:

1. **SMOTE** (Synthetic Minority Over-sampling Technique) -- applied to the training set only, before model fitting. This generates synthetic positive-class samples, balancing the training set to a 1:1 class ratio. Not applied to validation or test sets to preserve unbiased evaluation.

2. **`scale_pos_weight`** (Optuna-tuned to 11.44 pre-SMOTE, adjusted to 1.0 post-SMOTE) -- XGBoost's built-in class weighting. Since SMOTE already balances the training classes, `scale_pos_weight` is reset to neutral (1.0) to avoid over-compensating.

Using both provides complementary benefits: SMOTE enriches the feature space with realistic synthetic examples for training, while `scale_pos_weight` remains available as a tunable parameter if SMOTE is disabled or partial oversampling is used.

## Hyperparameter Tuning

XGBoost hyperparameters were optimized using **Optuna** (50 trials, 5-fold stratified cross-validation, AUC-ROC objective):

| Parameter | Tuned Value |
|-----------|-------------|
| n_estimators | 750 |
| max_depth | 4 |
| learning_rate | 0.0183 |
| subsample | 0.925 |
| colsample_bytree | 0.681 |
| min_child_weight | 6 |
| scale_pos_weight | 11.44 |

**Best CV AUC-ROC: 0.8668**

Tuned parameters are persisted in `models/best_params.json` and automatically loaded by the training pipeline.

## Model Performance

| Metric | Logistic Regression | XGBoost (Optuna-tuned) |
|--------|--------------------:|--------:|
| AUC-ROC | 0.789 | **0.848** |
| Precision | 0.175 | **0.419** |
| Recall | **0.648** | 0.389 |
| F1-Score | 0.276 | **0.403** |
| Brier Score | 0.161 | **0.063** |

## Explainability

SHAP (SHapley Additive exPlanations) provides two levels of transparency:

- **Global:** Which features matter most across all predictions (bar plot + beeswarm)
- **Per-applicant:** For each scoring request, the API computes real-time SHAP values using `TreeExplainer`, showing which features pushed that specific applicant's score up or down

Top features by global SHAP importance:
1. RevolvingUtilizationOfUnsecuredLines
2. TotalDelinquencies
3. age
4. DebtRatio
5. NumberOfTimes90DaysLate

### Counterfactual Explanations

For applicants who receive REVIEW or REJECT decisions, the system generates actionable suggestions showing the minimal feature changes that would flip the decision to APPROVE. Example: *"Increase MonthlyIncome from $3,000 to $4,500 (+$1,500)"*.

## Fairness Audit & Mitigation

Using age-based subgroups as proxy (no explicit protected attributes in the dataset):

### Before Mitigation (uniform 0.3 threshold)

| Metric | Value | Threshold | Status |
|--------|-------|-----------|--------|
| Disparate Impact (4/5 rule) | 0.69 | >= 0.80 | FAIL |
| Demographic Parity gap | 0.30 | <= 0.15 | FAIL |

Younger borrowers (18-30) had significantly lower approval rates than older borrowers (61+).

### After Mitigation (calibrated per-group thresholds)

| Age Group | Calibrated Threshold |
|-----------|--------------------:|
| 18-30 | 0.456 |
| 31-45 | 0.380 |
| 46-60 | 0.300 |
| 61+ | 0.300 |

| Metric | Value | Threshold | Status |
|--------|-------|-----------|--------|
| Disparate Impact (4/5 rule) | **0.856** | >= 0.80 | **PASS** |
| Demographic Parity gap | **0.140** | <= 0.15 | **PASS** |

Thresholds are computed via binary search per age group and stored in `models/fair_thresholds.json`. At inference time, the API automatically applies the group-specific threshold when `FAIRNESS_ENABLED=true` (default).

See `reports/fairness_report.md` for the full audit.

## Decision Thresholds

The system uses a three-tier decision structure with **fairness-calibrated, group-specific approval thresholds**:

| Default Probability | Decision |
|---------------------|----------|
| <= group threshold (0.30 - 0.46) | APPROVE |
| group threshold - 0.60 | REVIEW |
| > 0.60 | REJECT |

The F1-optimal threshold (0.42) was computed on the validation set and stored in model metadata for reference. The production thresholds are intentionally more conservative (lower) because in credit scoring, **false negatives (missed defaults) are costlier than false positives (denied good borrowers)**. The business-aligned thresholds at 0.30-0.46 prioritize recall of defaults over precision, which is standard practice in lending risk.

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
  "contribution_unit": "log-odds (positive = increases default risk)",
  "top_features": [
    {"feature": "RevolvingUtilizationOfUnsecuredLines", "value": 0.35, "contribution": 0.0821},
    {"feature": "TotalDelinquencies", "value": 1.0, "contribution": 0.0315}
  ]
}
```

### `POST /score/batch`

Score up to 100 applicants in a single request. Requires `Authorization: Bearer <token>` header.

**Request body:**
```json
{
  "applicants": [
    { ... applicant 1 ... },
    { ... applicant 2 ... }
  ]
}
```

**Response:**
```json
{
  "results": [ ... ScoreResponse objects ... ],
  "count": 2
}
```

### `POST /score/enriched`

Score an applicant using the enriched model (application + alternate data features from Home Credit). Requires `Authorization: Bearer <token>` header.

All 32 alternate data fields are optional -- when omitted, they default to 0.0 (matching how the model was trained for applicants without alternate data records).

**Request body:**
```json
{
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
  "inst_late_ratio": 0.1,
  "inst_avg_payment_ratio": 0.98,
  "cc_avg_utilization": 0.3,
  "bureau_count": 3,
  "bureau_overdue_ratio": 0.0
}
```

**Response:**
```json
{
  "model_version": "xgb_enriched",
  "score": 0.0892,
  "decision": "APPROVE",
  "contribution_unit": "log-odds (positive = increases default risk)",
  "top_features": [
    {"feature": "EXT_SOURCE_2", "value": 0.6, "contribution": -0.312},
    {"feature": "inst_late_ratio", "value": 0.1, "contribution": 0.087}
  ]
}
```

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.10+ |
| ML Model | XGBoost 2.0 (Optuna-tuned) |
| Baseline | scikit-learn (Logistic Regression) |
| Imbalance | SMOTE (imbalanced-learn) + scale_pos_weight |
| Explainability | SHAP 0.43 (per-request TreeExplainer) |
| Fairness | Custom threshold calibration (DI + parity) |
| Tuning | Optuna (50-trial Bayesian optimization) |
| API | FastAPI + Uvicorn |
| Dashboard | Streamlit |
| Validation | Pydantic 2.0 |
| Data | pandas, numpy, pyarrow |
| Visualization | matplotlib, seaborn |
| Testing | pytest (48 tests) |
| Containerization | Docker |
| CI | GitHub Actions |

## Alternate Data Integration

The hackathon theme emphasises **"leveraging alternative data sources"** for unbanked/under-banked populations. We go beyond the primary Give Me Some Credit dataset by integrating the [Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk) dataset -- a 2.7 GB, multi-table dataset specifically designed for credit scoring of applicants with little or no traditional credit history.

### Alternate Data Sources

Five transaction-level tables are ingested and aggregated per applicant via `src/alternate_data.py`:

| Source | Rows | Alternate Signal | Example Features |
|--------|-----:|-----------------|-----------------|
| **Instalment Payments** | 13.6M | Payment timeliness & completion | `inst_late_ratio`, `inst_avg_payment_ratio`, `inst_underpaid_ratio` |
| **Credit Card Balance** | 3.8M | Card utilisation & drawing patterns | `cc_avg_utilization`, `cc_avg_atm_drawings`, `cc_max_dpd` |
| **POS/Cash Balance** | 10.0M | Point-of-sale transaction behaviour | `pos_completed_ratio`, `pos_avg_dpd`, `pos_months` |
| **Previous Applications** | 1.7M | Loan application history | `prev_app_count`, `prev_approved_ratio`, `prev_refused_ratio` |
| **Bureau Records** | 1.7M | External credit bureau (partial) | `bureau_active_ratio`, `bureau_overdue_ratio`, `bureau_avg_debt` |

These signals are **real-world analogues** of the alternate data sources highlighted in the problem statement:
- **Transaction patterns** -- instalment payment behaviour and POS purchase records
- **Utility-like payments** -- regular instalment completion and timeliness
- **E-commerce / retail activity** -- POS loan records at retail stores
- **Financial behaviour signals** -- credit card usage patterns, ATM withdrawal frequency

### Impact: Alternate Data Improves Prediction

We ran a controlled experiment on the Home Credit dataset (307,511 applicants):

| Model | Features | AUC-ROC |
|-------|----------|---------|
| Application only | 17 traditional features | 0.750 |
| **Application + Alternate Data** | **17 + 32 alternate features** | **0.769** |

**Improvement: +0.019 AUC-ROC** from alternate data alone.

Top alternate data features by importance:

| Rank | Feature | Source | Signal |
|------|---------|--------|--------|
| 1 | `inst_underpaid_ratio` | Instalments | Fraction of instalments underpaid |
| 2 | `inst_max_delay` | Instalments | Maximum payment delay in days |
| 3 | `prev_refused_ratio` | Previous Apps | Fraction of prior loan applications refused |
| 4 | `inst_late_ratio` | Instalments | Fraction of instalments paid late |
| 5 | `prev_approved_ratio` | Previous Apps | Fraction of prior loan applications approved |

### Running the Alternate Data Pipeline

```bash
# Train the enriched model (saves to models/xgb_enriched.joblib)
python -m src.alternate_data
```

This loads all five alternate data tables, engineers 32 features, trains two XGBoost models (application-only baseline vs. enriched), and saves the enriched model for production serving via the `/score/enriched` endpoint.

### Production Integration Path

In a production deployment, the pipeline would ingest alternate data via:

1. **UPI / digital payment APIs** -- transaction frequency, average ticket size, merchant diversity (analogous to our instalment + POS features)
2. **Utility bill payment records** -- electricity, water, telecom payment timeliness (analogous to our `inst_late_ratio` and `inst_avg_payment_ratio`)
3. **Rent payment history** -- via landlord/property APIs or bank statement parsing (analogous to `inst_underpaid_ratio`)
4. **Mobile phone metadata** -- recharge frequency, data usage patterns (analogous to `DAYS_LAST_PHONE_CHANGE` in application data)
5. **E-commerce activity** -- purchase frequency, return rates, seller ratings for MSMEs (analogous to POS transaction records)

The `src/alternate_data.py` module is designed to be extensible: each `_engineer_*_features()` function follows the same pattern (load → aggregate per applicant → return fixed-width feature vector), making it straightforward to add new data sources.

## License

This project was built for the HackerEarth Hack-o-Hire 2026 hackathon.
