"""Fairness auditing utilities.

Computes subgroup-level metrics (AUC, disparate impact, demographic
parity) to detect potential bias.  Since the Give Me Some Credit dataset
contains no explicit protected attributes (gender, ethnicity, religion),
we use **age bins** as proxy subgroups -- a reasonable proxy because age
can correlate with credit access and is a protected characteristic under
many fair-lending regulations (e.g., ECOA).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.config import REPORTS_DIR, SCORE_THRESHOLDS
from src.models import load_model
from src.features import load_processed, TARGET_COL

FAIRNESS_REPORT_PATH = REPORTS_DIR / "fairness_report.md"

# Age bins used as proxy subgroups
AGE_BINS = [(18, 30, "18-30"), (31, 45, "31-45"), (46, 60, "46-60"), (61, 120, "61+")]

# Disparate impact threshold (4/5 rule)
DI_THRESHOLD = 0.80


def _bin_age(ages: pd.Series) -> pd.Series:
    """Assign age-group labels."""
    labels = pd.Series("unknown", index=ages.index)
    for lo, hi, lbl in AGE_BINS:
        mask = (ages >= lo) & (ages <= hi)
        labels[mask] = lbl
    return labels


def compute_subgroup_metrics(
    y_true: pd.Series,
    y_proba: np.ndarray,
    groups: pd.Series,
) -> pd.DataFrame:
    """Compute AUC, positive-prediction rate, and count per subgroup.

    Parameters
    ----------
    y_true : pd.Series
        True binary labels.
    y_proba : np.ndarray
        Predicted probabilities of the positive class.
    groups : pd.Series
        Group labels (same length as y_true).

    Returns
    -------
    pd.DataFrame
        One row per subgroup with columns: ``group``, ``count``,
        ``base_rate``, ``auc``, ``approval_rate``, ``avg_score``.
    """
    approve_threshold = SCORE_THRESHOLDS["APPROVE"]

    rows = []
    for g in sorted(groups.unique()):
        mask = groups == g
        yt = y_true[mask]
        yp = y_proba[mask]

        if len(yt) < 10 or yt.nunique() < 2:
            auc = float("nan")
        else:
            auc = roc_auc_score(yt, yp)

        approval_rate = float((yp <= approve_threshold).mean())
        rows.append({
            "group": g,
            "count": int(mask.sum()),
            "base_rate": float(yt.mean()),
            "auc": round(auc, 4) if not np.isnan(auc) else None,
            "approval_rate": round(approval_rate, 4),
            "avg_score": round(float(yp.mean()), 4),
        })

    return pd.DataFrame(rows)


def compute_disparate_impact(subgroup_df: pd.DataFrame) -> dict[str, Any]:
    """Compute disparate impact ratio from subgroup approval rates.

    Uses the 4/5 (80 %) rule: if any group's approval rate is less than
    80 % of the highest group's rate, there may be adverse impact.

    Returns
    -------
    dict
        ``best_group``, ``worst_group``, ``di_ratio``, ``pass``.
    """
    best_rate = subgroup_df["approval_rate"].max()
    worst_rate = subgroup_df["approval_rate"].min()
    best_group = subgroup_df.loc[subgroup_df["approval_rate"].idxmax(), "group"]
    worst_group = subgroup_df.loc[subgroup_df["approval_rate"].idxmin(), "group"]

    di_ratio = worst_rate / best_rate if best_rate > 0 else 0.0

    return {
        "best_group": best_group,
        "best_approval_rate": best_rate,
        "worst_group": worst_group,
        "worst_approval_rate": worst_rate,
        "di_ratio": round(di_ratio, 4),
        "pass": di_ratio >= DI_THRESHOLD,
    }


def compute_demographic_parity(subgroup_df: pd.DataFrame) -> dict[str, Any]:
    """Compute demographic parity gap (max difference in approval rates)."""
    rates = subgroup_df["approval_rate"]
    gap = float(rates.max() - rates.min())
    return {
        "max_approval_rate": float(rates.max()),
        "min_approval_rate": float(rates.min()),
        "parity_gap": round(gap, 4),
        "pass": gap <= 0.15,  # common threshold
    }


def _write_report(
    subgroup_df: pd.DataFrame,
    di_result: dict,
    dp_result: dict,
    output_path: Path | None = None,
) -> None:
    """Write the fairness report to Markdown."""
    output_path = output_path or FAIRNESS_REPORT_PATH
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# Fairness Report - InclusionScore AI",
        "",
        "## Methodology",
        "",
        "Since the *Give Me Some Credit* dataset contains **no explicit protected",
        "attributes** (gender, ethnicity, religion), we use **age groups** as proxy",
        "subgroups for this audit. Age is a protected characteristic under fair-lending",
        "regulations (e.g., ECOA) and correlates with credit access patterns.",
        "",
        "- **Approval** = predicted default probability <= "
        f"{SCORE_THRESHOLDS['APPROVE']}",
        f"- **Disparate impact threshold (4/5 rule)**: {DI_THRESHOLD}",
        "",
        "## Subgroup Metrics",
        "",
        subgroup_df.to_markdown(index=False),
        "",
        "## Disparate Impact Analysis",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Best group | {di_result['best_group']} "
        f"(approval rate {di_result['best_approval_rate']:.2%}) |",
        f"| Worst group | {di_result['worst_group']} "
        f"(approval rate {di_result['worst_approval_rate']:.2%}) |",
        f"| DI ratio | {di_result['di_ratio']:.4f} |",
        f"| **Result** | **{'PASS' if di_result['pass'] else 'FAIL'}** |",
        "",
        "## Demographic Parity",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Max approval rate | {dp_result['max_approval_rate']:.2%} |",
        f"| Min approval rate | {dp_result['min_approval_rate']:.2%} |",
        f"| Parity gap | {dp_result['parity_gap']:.4f} |",
        f"| **Result** | **{'PASS' if dp_result['pass'] else 'FAIL'}** |",
        "",
        "## Mitigation Recommendations",
        "",
    ]

    if not di_result["pass"] or not dp_result["pass"]:
        lines.extend([
            "Bias detected. Recommended mitigations:",
            "",
            "1. **Re-calibrate thresholds** per subgroup to equalise approval rates.",
            "2. **Apply in-processing** techniques (e.g., fairness-aware regularisation).",
            "3. **Post-processing** adjustments: shift predicted probabilities per group",
            "   so that approval rates satisfy the 4/5 rule.",
            "4. **Collect additional features** that reduce reliance on age as a proxy.",
            "",
        ])
    else:
        lines.extend([
            "No significant bias detected under the 4/5 rule or demographic parity",
            "threshold. Continue monitoring with fresh data.",
            "",
        ])

    lines.extend([
        "## Notes",
        "",
        "- This audit uses age-based subgroups only. A production system should",
        "  audit additional protected attributes if available.",
        "- Results depend on the approval threshold; changing SCORE_THRESHOLDS",
        "  will change approval rates and fairness metrics.",
        "",
    ])

    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Fairness report written to {output_path}")


def run_audit(
    model_path: str | Path,
    data_path: str | Path,
    *,
    output_path: Path | None = None,
) -> dict[str, Any]:
    """Run the full fairness audit pipeline.

    Parameters
    ----------
    model_path : str | Path
        Path to the saved model (``.joblib``).
    data_path : str | Path
        Path to the test data (Parquet or CSV).
    output_path : Path | None
        Where to write the Markdown report.

    Returns
    -------
    dict
        Audit results including subgroup metrics, disparate impact,
        and demographic parity.
    """
    model = load_model(model_path)

    if str(data_path).endswith(".parquet"):
        df = pd.read_parquet(data_path)
    else:
        df = pd.read_csv(data_path)

    y_true = df.pop(TARGET_COL)
    X = df
    y_proba = model.predict_proba(X)[:, 1]

    # Create age-based subgroups
    groups = _bin_age(X["age"])

    subgroup_df = compute_subgroup_metrics(y_true, y_proba, groups)
    di_result = compute_disparate_impact(subgroup_df)
    dp_result = compute_demographic_parity(subgroup_df)

    _write_report(subgroup_df, di_result, dp_result, output_path)

    # Also save a subgroup AUC bar chart
    fig, ax = plt.subplots(figsize=(7, 4))
    valid = subgroup_df.dropna(subset=["auc"])
    ax.bar(valid["group"], valid["auc"], color="#2196F3")
    ax.set_ylim(0.5, 1.0)
    ax.set_ylabel("AUC-ROC")
    ax.set_title("Subgroup AUC by Age Group")
    for i, row in valid.iterrows():
        ax.text(i, row["auc"] + 0.005, f'{row["auc"]:.3f}', ha="center", fontsize=9)
    plt.tight_layout()
    chart_dir = REPORTS_DIR / "explainability"
    chart_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(chart_dir / "subgroup_auc.png", dpi=150)
    plt.close()
    print(f"Saved {chart_dir / 'subgroup_auc.png'}")

    return {
        "subgroup_metrics": subgroup_df.to_dict(orient="records"),
        "disparate_impact": di_result,
        "demographic_parity": dp_result,
    }
