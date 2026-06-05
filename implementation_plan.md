# InclusionScore-AI — Full Audit Report & Implementation Plan

## Audit Summary

Audited all 14 source files, 4 test files, model metadata, and configs. **All 48 existing tests pass.** The project is well-structured overall, but has one major bug and several UI/UX improvements needed.

---

## 🔴 Critical Bug: Enriched Model Scoring Shows Same Output

### Root Cause

In [dashboard.py](file:///c:/Users/cheta/Desktop/FOLDERS/Hackathon/InclusionScore-AI/app/dashboard.py#L232-L258), the enriched model [EnrichedApplicantInput](file:///c:/Users/cheta/Desktop/FOLDERS/Hackathon/InclusionScore-AI/src/api_enriched.py#56-127) is built with **6 fields hardcoded** to constant values, never connected to any user input:

```python
# Lines 240-248 — these NEVER change regardless of form input
DAYS_REGISTRATION=-12563,      # hardcoded
DAYS_ID_PUBLISH=-4260,         # hardcoded
CNT_CHILDREN=0,                # hardcoded
CNT_FAM_MEMBERS=2,             # hardcoded
REGION_RATING_CLIENT=2,        # hardcoded
DAYS_LAST_PHONE_CHANGE=-1134,  # hardcoded
```

Additionally, the model expects **49 features** total (17 app + 32 alt-data from [metadata](file:///c:/Users/cheta/Desktop/FOLDERS/Hackathon/InclusionScore-AI/models/xgb_enriched_metadata.json)), but the dashboard form only exposes **18 fields** (9 app + 9 alt-data). The remaining **31 fields** silently default to `None → 0.0` via [prepare_enriched_input()](file:///c:/Users/cheta/Desktop/FOLDERS/Hackathon/InclusionScore-AI/src/api_enriched.py#140-158). 

This means **the dominant features** (`EXT_SOURCE_*`, `DAYS_BIRTH`, `DAYS_EMPLOYED`) are indeed connected to the form — but several other impactful features (`OBS_30_CNT_SOCIAL_CIRCLE`, `DEF_30_CNT_SOCIAL_CIRCLE`, many alt-data fields) are always zero. The output **does change** but only significantly when you vary the 3 `EXT_SOURCE_*` fields or `DAYS_BIRTH`/`DAYS_EMPLOYED`. Most of the alt-data fields the user can change have **near-zero feature importance**, so they barely move the score.

### Why It Feels "Same Output"

The fields exposed in the form have low model importance. The fields with high importance are either:
- Connected but with defaults that always land in the same decision region
- Hardcoded and never change

---

## Proposed Changes

### 1. Fix Enriched Model Form — Expose All Important Features

#### [MODIFY] [dashboard.py](file:///c:/Users/cheta/Desktop/FOLDERS/Hackathon/InclusionScore-AI/app/dashboard.py)

Rewrite the Enriched Model Scoring section (lines 194-284) to:

1. **Replace hardcoded fields** with `st.number_input()` controls for all 6 currently-hardcoded fields
2. **Add missing alt-data fields** — expose the remaining ~22 alternate data features that currently default to 0.0, organized in grouped expanders:
   - Installment Features (7 fields — currently only 3 exposed)
   - Credit Card Features (9 fields — currently only 1 exposed) 
   - POS/Cash Features (5 fields — currently 1 exposed)
   - Previous Applications (5 fields — currently 2 exposed)
   - Bureau Features (6 fields — currently 2 exposed)
3. **Use sensible defaults** matching the training data medians instead of 0.0, so the initial baseline score is more realistic
4. **Add `st.session_state` tracking** so results persist across reruns (Streamlit re-executes the script on every interaction)

### 2. UI/UX Improvements

#### [MODIFY] [dashboard.py](file:///c:/Users/cheta/Desktop/FOLDERS/Hackathon/InclusionScore-AI/app/dashboard.py)

- **Add color-coded decision banners** with `st.success`/`st.warning`/`st.error` instead of plain `st.metric` for clearer visual feedback
- **Move enriched scoring into a tab** instead of being appended below the primary scoring, using `st.tabs(["Primary Scoring", "Enriched Model Scoring"])` to reduce scrolling and organize the two scoring modes
- **Add a "Reset to Defaults" button** in the enriched form
- **Add tooltips** to all enriched form fields explaining what each feature means in plain English
- **Fix the primary scoring section** to also use session state so results don't disappear when scrolling

### 3. Minor Code Quality Fixes

#### [MODIFY] [api_enriched.py](file:///c:/Users/cheta/Desktop/FOLDERS/Hackathon/InclusionScore-AI/src/api_enriched.py)

- No code changes needed — the scoring logic is correct. The bug is entirely in the dashboard's form wiring.

---

## No Changes Needed (Audit Passed ✅)

| Component | Status |
|-----------|--------|
| [src/api.py](file:///c:/Users/cheta/Desktop/FOLDERS/Hackathon/InclusionScore-AI/src/api.py) — Primary scoring API | ✅ Correct |
| [src/config.py](file:///c:/Users/cheta/Desktop/FOLDERS/Hackathon/InclusionScore-AI/src/config.py) — Configuration | ✅ Correct |
| [src/models.py](file:///c:/Users/cheta/Desktop/FOLDERS/Hackathon/InclusionScore-AI/src/models.py) — Model loading/saving | ✅ Correct |
| [src/features.py](file:///c:/Users/cheta/Desktop/FOLDERS/Hackathon/InclusionScore-AI/src/features.py) — Feature engineering | ✅ Correct |
| [src/fairness.py](file:///c:/Users/cheta/Desktop/FOLDERS/Hackathon/InclusionScore-AI/src/fairness.py) — Fairness auditing | ✅ Correct |
| [src/counterfactual.py](file:///c:/Users/cheta/Desktop/FOLDERS/Hackathon/InclusionScore-AI/src/counterfactual.py) — Counterfactual explanations | ✅ Correct |
| [src/explainability.py](file:///c:/Users/cheta/Desktop/FOLDERS/Hackathon/InclusionScore-AI/src/explainability.py) — SHAP utilities | ✅ Correct |
| [src/alternate_data.py](file:///c:/Users/cheta/Desktop/FOLDERS/Hackathon/InclusionScore-AI/src/alternate_data.py) — Alternate data pipeline | ✅ Correct |
| [api/main.py](file:///c:/Users/cheta/Desktop/FOLDERS/Hackathon/InclusionScore-AI/api/main.py) — FastAPI entrypoint | ✅ Correct |
| `tests/` — All 48 tests | ✅ All Passing |
| Model artifacts (`xgb_v1`, `xgb_enriched`) | ✅ Present & valid |

---

## Verification Plan

### Automated Tests
```bash
python -m pytest tests/ -v --tb=short
```
All 48 existing tests should continue to pass. The existing test [test_enriched_different_applicants_different_shap](file:///c:/Users/cheta/Desktop/FOLDERS/Hackathon/InclusionScore-AI/tests/test_api.py#244-262) in [tests/test_api.py](file:///c:/Users/cheta/Desktop/FOLDERS/Hackathon/InclusionScore-AI/tests/test_api.py) already validates that the API-level enriched scoring produces different outputs for different inputs — confirming the bug is in the dashboard wiring only.

### Manual Verification
After the fix, the user should:
1. Start the dashboard: `streamlit run app/dashboard.py`
2. Go to the **Enriched Model Scoring** tab
3. Click "Score with Enriched Model" with default values → note the score
4. Change `EXT_SOURCE_2` from `0.6` to `0.1` → click score again → score should **increase significantly** (higher default risk)
5. Change `Late Payment Ratio` from `0.1` to `0.9` → score should increase
6. Change `Total Income` from `270000` to `50000` → score should increase
7. Verify that all new form fields are visible and properly organized
