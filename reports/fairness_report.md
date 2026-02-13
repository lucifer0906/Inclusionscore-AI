# Fairness Report - InclusionScore AI

## Methodology

Since the *Give Me Some Credit* dataset contains **no explicit protected
attributes** (gender, ethnicity, religion), we use **age groups** as proxy
subgroups for this audit. Age is a protected characteristic under fair-lending
regulations (e.g., ECOA) and correlates with credit access patterns.

- **Approval** = predicted default probability <= approval threshold
- **Disparate impact threshold (4/5 rule)**: 0.80
- **Demographic parity threshold**: parity gap <= 0.15

## Subgroup Metrics (Test Set, n = 22,500)

| Group | Count | Base Rate | AUC | Approval Rate | Avg Score |
|:------|------:|----------:|----:|-------------:|----------:|
| 18-30 | 1,566 | 11.4% | 0.788 | 67.0% | 0.260 |
| 31-45 | 6,104 | 9.7% | 0.820 | 75.8% | 0.214 |
| 46-60 | 7,974 | 6.7% | 0.849 | 86.0% | 0.153 |
| 61+ | 6,856 | 2.9% | 0.851 | 97.2% | 0.057 |

**Observation:** Younger borrowers (18-30) have a higher base default rate *and*
a lower approval rate. The approval rate gap between the youngest and oldest
groups is ~30 percentage points under a uniform threshold.

---

## Before Mitigation (Uniform Threshold = 0.3)

All age groups are evaluated with the same approval threshold of 0.30.

### Disparate Impact Analysis

| Metric | Value |
|--------|-------|
| Best group | 61+ (approval rate 97.21%) |
| Worst group | 18-30 (approval rate 66.99%) |
| DI ratio | 0.6891 |
| **Result** | **FAIL** (threshold: >= 0.80) |

### Demographic Parity

| Metric | Value |
|--------|-------|
| Max approval rate | 97.21% |
| Min approval rate | 66.99% |
| Parity gap | 0.3022 |
| **Result** | **FAIL** (threshold: <= 0.15) |

**Conclusion:** Under a uniform threshold, the system exhibits significant
age-based disparate impact. Younger applicants are disproportionately denied.

---

## Mitigation: Per-Group Threshold Calibration

### Approach

We use **binary search per age group** to find the minimum approval threshold
for each group such that:

1. **Disparate impact (4/5 rule):** every group's approval rate is >= 80% of
   the best group's rate
2. **Demographic parity:** the gap between the highest and lowest approval
   rates is <= 0.15

The algorithm:
- Start with the base threshold (0.30) for the best-performing group
- For under-performing groups, raise the threshold (accept higher default
  probabilities) until the approval rate meets the target
- Binary search converges in ~100 iterations per group

### Calibrated Thresholds

| Age Group | Calibrated Threshold | Effect |
|-----------|--------------------:|--------|
| 18-30 | 0.456 | Raised from 0.30 -- more lenient for youngest group |
| 31-45 | 0.380 | Raised from 0.30 |
| 46-60 | 0.300 | Unchanged |
| 61+ | 0.300 | Unchanged (reference group) |

These thresholds are stored in `models/fair_thresholds.json` and loaded at
inference time when `FAIRNESS_ENABLED=true` (default).

---

## After Mitigation (Calibrated Per-Group Thresholds)

### Disparate Impact Analysis

| Metric | Value |
|--------|-------|
| Best group | 61+ |
| Worst group | 31-45 |
| DI ratio | **0.8561** |
| **Result** | **PASS** (>= 0.80) |

### Demographic Parity

| Metric | Value |
|--------|-------|
| Max approval rate | 97.21% |
| Min approval rate | 83.22% |
| Parity gap | **0.1399** |
| **Result** | **PASS** (<= 0.15) |

### Summary

| Metric | Before | After | Status |
|--------|--------|-------|--------|
| Disparate Impact | 0.6891 | **0.8561** | FAIL -> **PASS** |
| Parity Gap | 0.3022 | **0.1399** | FAIL -> **PASS** |

---

## Trade-off Analysis

Raising thresholds for younger groups increases their approval rate but also
increases the expected default rate among approved applicants in those groups.
This is a deliberate business trade-off: accepting slightly higher risk in
historically under-served groups in exchange for equitable access to credit.

In production, this trade-off should be monitored with:
- **Group-level default tracking** post-deployment
- **Periodic recalibration** as population distributions shift
- **Regulatory review** to ensure compliance with ECOA and fair lending guidelines

## Notes

- This audit uses age-based subgroups only. A production system should
  audit additional protected attributes if available.
- Results depend on the approval threshold; changing SCORE_THRESHOLDS
  will change approval rates and fairness metrics.
- The calibration is statistically derived and reproducible -- see
  `src/fairness.py:calibrate_fair_thresholds()` for the implementation.
