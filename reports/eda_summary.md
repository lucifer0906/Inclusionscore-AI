# EDA Summary - InclusionScore AI

## Dataset

- **Source:** Give Me Some Credit (Kaggle) - `cs-training.csv`
- **Rows:** 150,000
- **Columns:** 11 (10 features + 1 target)
- **Target column:** `SeriousDlqin2yrs` (binary: 1 = experienced 90+ days past due delinquency or worse within 2 years)

## Column List

| Column | Dtype | Description |
|--------|-------|-------------|
| `SeriousDlqin2yrs` | int64 | **Target.** 90+ days delinquency flag |
| `RevolvingUtilizationOfUnsecuredLines` | float64 | Total balance on revolving credit / credit limit |
| `age` | int64 | Age of borrower |
| `NumberOfTime30-59DaysPastDueNotWorse` | int64 | Number of times 30-59 days past due |
| `DebtRatio` | float64 | Monthly debt payments / gross monthly income |
| `MonthlyIncome` | float64 | Monthly income |
| `NumberOfOpenCreditLinesAndLoans` | int64 | Number of open loans + credit lines |
| `NumberOfTimes90DaysLate` | int64 | Number of times 90+ days late |
| `NumberRealEstateLoansOrLines` | int64 | Number of mortgage/RE loans |
| `NumberOfTime60-89DaysPastDueNotWorse` | int64 | Number of times 60-89 days past due |
| `NumberOfDependents` | float64 | Number of dependents |

## Target Imbalance

- **No Default (0):** 139,974 (93.3%)
- **Default (1):** 10,026 (6.7%)
- **Imbalance ratio:** ~1:14
- **Action:** Dual strategy implemented -- SMOTE (applied to training set only, before model fitting) + `scale_pos_weight` (Optuna-tuned to 11.44) in XGBoost loss function.

## Missing Values

| Column | Missing Count | Missing % | Imputation Strategy |
|--------|--------------|-----------|-------------------|
| `MonthlyIncome` | 29,731 | 19.8% | Median imputation (median = 5,400) |
| `NumberOfDependents` | 3,924 | 2.6% | Mode imputation (mode = 0) |

All other columns have zero missing values.

## Outliers & Anomalies

| Issue | Count | Action |
|-------|-------|--------|
| `age == 0` | 1 row | Drop (invalid) |
| `RevolvingUtilizationOfUnsecuredLines > 1` | 3,321 rows | Cap at 1.0 (utilization > 100% is anomalous) |
| `DebtRatio > 10` | 28,877 rows | Cap at 99th percentile (~3,517) |
| Delinquency cols = 96 or 98 | 269 rows | Sentinel/error values; cap at 15 |

## Top Correlations with Target

| Feature | Correlation |
|---------|------------|
| `NumberOfTime30-59DaysPastDueNotWorse` | +0.126 |
| `NumberOfTimes90DaysLate` | +0.117 |
| `NumberOfTime60-89DaysPastDueNotWorse` | +0.102 |
| `age` | -0.115 |
| `NumberOfOpenCreditLinesAndLoans` | -0.030 |
| `MonthlyIncome` | -0.020 |

Key observations:
- Past-due delinquency counts are the strongest positive predictors of default.
- Younger borrowers are more likely to default.
- Linear correlations are modest, suggesting non-linear models (XGBoost) will capture complex interactions better.

## Cleaning Plan (for Phase 2)

1. **Drop:** Rows where `age == 0` (1 row).
2. **Impute:** `MonthlyIncome` with median, `NumberOfDependents` with mode (0).
3. **Cap outliers:**
   - `RevolvingUtilizationOfUnsecuredLines` clipped to [0, 1]
   - `DebtRatio` clipped at 99th percentile
   - Three delinquency columns: replace values >= 96 with cap value of 15
4. **No protected attributes** to remove (dataset has no gender/ethnicity/religion columns).

## Engineered Features (for Phase 2)

| Feature | Formula | Rationale |
|---------|---------|-----------|
| `TotalDelinquencies` | Sum of 3 past-due columns | Aggregate delinquency signal |
| `HasDelinquency` | 1 if any past-due > 0 | Binary delinquency flag |
| `IncomeToDebtRatio` | `MonthlyIncome / (DebtRatio + 1)` | Affordability proxy |
| `AgeBin` | Binned age (18-30, 31-45, 46-60, 61+) | Capture non-linear age effects |
| `CreditUtilBucket` | Binned revolving utilization | Segment utilization behavior |
| `OpenCreditPerDependent` | `OpenCreditLines / (Dependents + 1)` | Credit load per household |

## EDA Plots

All saved in `reports/eda_plots/`:
- `target_distribution.png` - Target class bar chart
- `missing_values.png` - Horizontal bar chart of missing %
- `correlation_matrix.png` - Lower-triangle heatmap
- `age_distribution.png` - Overlaid histograms by target
- `boxplots_by_target.png` - 6 feature boxplots by default status
- `delinquency_distributions.png` - Bar charts of past-due columns
