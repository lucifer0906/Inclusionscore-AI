"""Counterfactual explanations for rejected / review applicants.

For a given applicant whose decision is not APPROVE, finds minimal
feature changes that would flip the decision to APPROVE.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.api import (
    ApplicantInput,
    _get_model,
    _make_decision,
    prepare_input,
)

# Features we allow the counterfactual search to modify (mutable + sensible)
MUTABLE_FEATURES = [
    ("MonthlyIncome", "increase", 500, 50_000),
    ("DebtRatio", "decrease", 0.0, None),
    ("RevolvingUtilizationOfUnsecuredLines", "decrease", 0.0, None),
    ("NumberOfOpenCreditLinesAndLoans", "increase", 0, 30),
]


def generate_counterfactuals(
    applicant: ApplicantInput,
    current_score: float,
    current_decision: str,
    *,
    max_suggestions: int = 3,
) -> list[dict]:
    """Generate actionable counterfactual suggestions.

    Parameters
    ----------
    applicant : ApplicantInput
        Original applicant data.
    current_score : float
        Current predicted default probability.
    current_decision : str
        Current decision (APPROVE / REVIEW / REJECT).
    max_suggestions : int
        Max number of suggestions to return.

    Returns
    -------
    list[dict]
        Each dict has ``feature``, ``current_value``, ``suggested_value``,
        ``change``, and ``description``.
    """
    if current_decision == "APPROVE":
        return []

    model = _get_model()
    suggestions = []

    for feat_name, direction, bound, limit in MUTABLE_FEATURES:
        # Get current raw value
        raw_val = getattr(applicant, feat_name, None)
        if raw_val is None:
            continue

        # Try incremental changes to find the flip point
        best_new_val = None
        if direction == "increase":
            step = (limit - raw_val) / 20 if limit else raw_val * 0.1
            if step <= 0:
                continue
            for mult in range(1, 21):
                test_val = raw_val + step * mult
                if limit and test_val > limit:
                    break
                new_applicant = applicant.model_copy(update={feat_name: test_val})
                X = prepare_input(new_applicant)
                new_prob = float(model.predict_proba(X)[0, 1])
                new_decision = _make_decision(new_prob, age=applicant.age)
                if new_decision == "APPROVE":
                    best_new_val = round(test_val, 2)
                    break
        elif direction == "decrease":
            current = raw_val
            if current <= bound:
                continue
            step = (current - bound) / 20
            if step <= 0:
                continue
            for mult in range(1, 21):
                test_val = current - step * mult
                if test_val < bound:
                    test_val = bound
                new_applicant = applicant.model_copy(update={feat_name: test_val})
                X = prepare_input(new_applicant)
                new_prob = float(model.predict_proba(X)[0, 1])
                new_decision = _make_decision(new_prob, age=applicant.age)
                if new_decision == "APPROVE":
                    best_new_val = round(test_val, 2)
                    break

        if best_new_val is not None:
            change = best_new_val - raw_val
            if direction == "increase":
                desc = f"Increase {feat_name} from {raw_val:.2f} to {best_new_val:.2f} (+{change:.2f})"
            else:
                desc = f"Decrease {feat_name} from {raw_val:.2f} to {best_new_val:.2f} ({change:.2f})"
            suggestions.append({
                "feature": feat_name,
                "current_value": round(raw_val, 4),
                "suggested_value": best_new_val,
                "change": round(change, 4),
                "description": desc,
            })

        if len(suggestions) >= max_suggestions:
            break

    return suggestions
