# Fairness Report - InclusionScore AI

## Methodology

Since the *Give Me Some Credit* dataset contains **no explicit protected
attributes** (gender, ethnicity, religion), we use **age groups** as proxy
subgroups for this audit. Age is a protected characteristic under fair-lending
regulations (e.g., ECOA) and correlates with credit access patterns.

- **Approval** = predicted default probability <= 0.3
- **Disparate impact threshold (4/5 rule)**: 0.8

## Subgroup Metrics

| group   |   count |   base_rate |    auc |   approval_rate |   avg_score |
|:--------|--------:|------------:|-------:|----------------:|------------:|
| 18-30   |    1566 |   0.113665  | 0.7877 |          0.6699 |      0.2597 |
| 31-45   |    6104 |   0.0969856 | 0.8203 |          0.7577 |      0.2138 |
| 46-60   |    7974 |   0.0672185 | 0.8493 |          0.86   |      0.1531 |
| 61+     |    6856 |   0.0288798 | 0.8513 |          0.9721 |      0.0574 |

## Disparate Impact Analysis

| Metric | Value |
|--------|-------|
| Best group | 61+ (approval rate 97.21%) |
| Worst group | 18-30 (approval rate 66.99%) |
| DI ratio | 0.6891 |
| **Result** | **FAIL** |

## Demographic Parity

| Metric | Value |
|--------|-------|
| Max approval rate | 97.21% |
| Min approval rate | 66.99% |
| Parity gap | 0.3022 |
| **Result** | **FAIL** |

## Mitigation Recommendations

Bias detected. Recommended mitigations:

1. **Re-calibrate thresholds** per subgroup to equalise approval rates.
2. **Apply in-processing** techniques (e.g., fairness-aware regularisation).
3. **Post-processing** adjustments: shift predicted probabilities per group
   so that approval rates satisfy the 4/5 rule.
4. **Collect additional features** that reduce reliance on age as a proxy.

## Notes

- This audit uses age-based subgroups only. A production system should
  audit additional protected attributes if available.
- Results depend on the approval threshold; changing SCORE_THRESHOLDS
  will change approval rates and fairness metrics.
