"""Streamlit dashboard for InclusionScore AI.

Interactive form for applicant data entry with real-time scoring,
SHAP waterfall visualisation, and counterfactual suggestions.

Run:  streamlit run app/dashboard.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is on sys.path for src imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
import pandas as pd
import numpy as np

from src.api import ApplicantInput, score as score_applicant
from src.counterfactual import generate_counterfactuals

# Try to import enriched model scoring (optional — requires trained enriched model)
try:
    from src.api_enriched import EnrichedApplicantInput, score_enriched
    _ENRICHED_AVAILABLE = Path(PROJECT_ROOT / "models" / "xgb_enriched.joblib").exists()
except ImportError:
    _ENRICHED_AVAILABLE = False

st.set_page_config(
    page_title="InclusionScore AI",
    page_icon="🏦",
    layout="wide",
)

st.title("InclusionScore AI")
st.markdown("**AI-powered alternate credit scoring for the unbanked & under-banked**")
st.divider()

# ── Sidebar: applicant input form ────────────────────────────────────────
with st.sidebar:
    st.header("Applicant Information")

    age = st.number_input("Age", min_value=18, max_value=120, value=35)
    monthly_income = st.number_input(
        "Monthly Income ($)", min_value=0.0, value=5000.0, step=100.0
    )
    debt_ratio = st.number_input(
        "Debt Ratio (monthly debt / income)", min_value=0.0, value=0.3, step=0.01,
        format="%.4f",
    )
    revolving_util = st.number_input(
        "Revolving Utilization", min_value=0.0, value=0.5, step=0.01,
        format="%.4f",
        help="Total revolving balance / credit limit",
    )
    open_credit_lines = st.number_input(
        "Open Credit Lines & Loans", min_value=0, value=8
    )
    real_estate_loans = st.number_input(
        "Real Estate Loans", min_value=0, value=1
    )
    dependents = st.number_input("Number of Dependents", min_value=0, value=1)

    st.subheader("Delinquency History")
    times_30_59 = st.number_input(
        "Times 30-59 Days Past Due", min_value=0, value=0
    )
    times_60_89 = st.number_input(
        "Times 60-89 Days Past Due", min_value=0, value=0
    )
    times_90_plus = st.number_input(
        "Times 90+ Days Late", min_value=0, value=0
    )

    submit = st.button("Score Applicant", type="primary", use_container_width=True)

# ── Main panel: results ──────────────────────────────────────────────────
if submit:
    applicant = ApplicantInput(
        RevolvingUtilizationOfUnsecuredLines=revolving_util,
        age=age,
        **{"NumberOfTime30-59DaysPastDueNotWorse": times_30_59},
        DebtRatio=debt_ratio,
        MonthlyIncome=monthly_income,
        NumberOfOpenCreditLinesAndLoans=open_credit_lines,
        NumberOfTimes90DaysLate=times_90_plus,
        NumberRealEstateLoansOrLines=real_estate_loans,
        **{"NumberOfTime60-89DaysPastDueNotWorse": times_60_89},
        NumberOfDependents=float(dependents),
    )

    with st.spinner("Running model..."):
        result = score_applicant(applicant)

    # Decision banner
    color_map = {"APPROVE": "green", "REVIEW": "orange", "REJECT": "red"}
    icon_map = {"APPROVE": "✅", "REVIEW": "⚠️", "REJECT": "❌"}
    decision = result.decision
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Default Probability", f"{result.score:.4f}")
    with col2:
        st.metric("Decision", f"{icon_map.get(decision, '')} {decision}")
    with col3:
        st.metric("Model Version", result.model_version)

    st.divider()

    # SHAP contributions
    st.subheader("Top Feature Contributions (SHAP)")
    contrib_data = pd.DataFrame([
        {
            "Feature": fc.feature,
            "Value": round(fc.value, 4),
            "Contribution": round(fc.contribution, 4),
        }
        for fc in result.top_features
    ])

    if not contrib_data.empty:
        # Horizontal bar chart
        chart_data = contrib_data.set_index("Feature")["Contribution"].sort_values()
        colors = ["#e74c3c" if v > 0 else "#2ecc71" for v in chart_data.values]
        st.bar_chart(chart_data, horizontal=True)
        st.dataframe(contrib_data, use_container_width=True, hide_index=True)

    st.divider()

    # Counterfactual explanations
    if decision != "APPROVE":
        st.subheader("How to Improve Your Score")
        st.markdown(
            "The following changes could help move the decision to **APPROVE**:"
        )
        suggestions = generate_counterfactuals(
            applicant, result.score, decision
        )
        if suggestions:
            for i, s in enumerate(suggestions, 1):
                st.info(f"**Suggestion {i}:** {s['description']}")
        else:
            st.warning(
                "No simple single-feature change found to flip the decision. "
                "Multiple factors may need improvement simultaneously."
            )
    else:
        st.success("This applicant qualifies for approval. No changes needed.")

    st.divider()
    st.caption(
        "InclusionScore AI uses SHAP (SHapley Additive exPlanations) for "
        "per-applicant explainability and fairness-calibrated thresholds "
        "across age groups to ensure equitable outcomes."
    )
else:
    st.info("Fill in the applicant details in the sidebar and click **Score Applicant**.")
    st.markdown("""
    ### Features
    - **Real-time scoring** with XGBoost model (Optuna-tuned)
    - **Per-applicant SHAP explanations** showing which factors drive the decision
    - **Fairness-calibrated thresholds** ensuring equitable treatment across age groups
    - **Counterfactual suggestions** for rejected applicants showing how to improve
    """)

    # Alternate data integration section
    st.divider()
    st.subheader("Alternate Data Integration")
    st.markdown("""
    InclusionScore AI integrates **5 alternate data tables** from the
    Home Credit Default Risk dataset (30M+ transaction records) to
    score applicants with little or no traditional credit history.

    | Source | Signal | Example Features |
    |--------|--------|-----------------|
    | **Instalment Payments** (13.6M rows) | Payment timeliness & completion | Late-payment ratio, underpayment ratio |
    | **Credit Card Balance** (3.8M rows) | Card utilisation patterns | Avg utilisation, ATM drawing frequency |
    | **POS/Cash Balance** (10.0M rows) | Point-of-sale transactions | Completion ratio, days past due |
    | **Previous Applications** (1.7M rows) | Loan application history | Approval ratio, previous credit amounts |
    | **Bureau Records** (1.7M rows) | External credit bureau | Active credit ratio, overdue amounts |

    **Impact:** Adding 32 alternate data features improves AUC-ROC by
    **+0.019** (0.750 -> 0.769) on the Home Credit dataset,
    demonstrating that transaction patterns and payment behaviour
    provide meaningful creditworthiness signals for unbanked populations.

    Run `python -m src.alternate_data` to reproduce this experiment.
    """)


# ── Enriched Model Scoring Tab ──────────────────────────────────────────
st.divider()
if _ENRICHED_AVAILABLE:
    st.subheader("Enriched Model Scoring (Home Credit + Alternate Data)")
    st.markdown(
        "Score applicants using the **enriched model** trained on "
        "application features plus 32 alternate data features from "
        "5 Home Credit transaction tables."
    )

    with st.expander("Enriched Model Input Form", expanded=False):
        ecol1, ecol2 = st.columns(2)
        with ecol1:
            st.markdown("**Application Features**")
            e_income = st.number_input("Total Income", value=270000.0, key="e_income")
            e_credit = st.number_input("Credit Amount", value=1293502.5, key="e_credit")
            e_annuity = st.number_input("Annuity", value=35698.5, key="e_annuity")
            e_goods = st.number_input("Goods Price", value=1129500.0, key="e_goods")
            e_birth = st.number_input("Days Birth (negative)", value=-12005, key="e_birth")
            e_employed = st.number_input("Days Employed (negative)", value=-4542, key="e_employed")
            e_ext1 = st.number_input("External Source 1", value=0.5, key="e_ext1", format="%.3f")
            e_ext2 = st.number_input("External Source 2", value=0.6, key="e_ext2", format="%.3f")
            e_ext3 = st.number_input("External Source 3", value=0.55, key="e_ext3", format="%.3f")

        with ecol2:
            st.markdown("**Alternate Data Features**")
            e_inst_late = st.number_input("Late Payment Ratio", value=0.1, key="e_inst_late", format="%.3f")
            e_inst_pay_ratio = st.number_input("Avg Payment Ratio", value=0.98, key="e_inst_pay", format="%.3f")
            e_inst_underpaid = st.number_input("Underpaid Ratio", value=0.05, key="e_inst_under", format="%.3f")
            e_cc_util = st.number_input("Credit Card Utilization", value=0.3, key="e_cc_util", format="%.3f")
            e_bureau_count = st.number_input("Bureau Record Count", value=3, key="e_bureau_ct")
            e_bureau_overdue = st.number_input("Bureau Overdue Ratio", value=0.0, key="e_bureau_od", format="%.3f")
            e_prev_count = st.number_input("Previous App Count", value=5, key="e_prev_ct")
            e_prev_refused = st.number_input("Previous Refused Ratio", value=0.1, key="e_prev_ref", format="%.3f")
            e_pos_dpd = st.number_input("POS Max Days Past Due", value=0, key="e_pos_dpd")

        enriched_submit = st.button("Score with Enriched Model", type="primary", key="enrich_btn")

    if enriched_submit:
        enriched_input = EnrichedApplicantInput(
            AMT_INCOME_TOTAL=e_income,
            AMT_CREDIT=e_credit,
            AMT_ANNUITY=e_annuity,
            AMT_GOODS_PRICE=e_goods,
            DAYS_BIRTH=e_birth,
            DAYS_EMPLOYED=e_employed,
            DAYS_REGISTRATION=-12563,
            DAYS_ID_PUBLISH=-4260,
            EXT_SOURCE_1=e_ext1,
            EXT_SOURCE_2=e_ext2,
            EXT_SOURCE_3=e_ext3,
            CNT_CHILDREN=0,
            CNT_FAM_MEMBERS=2,
            REGION_RATING_CLIENT=2,
            DAYS_LAST_PHONE_CHANGE=-1134,
            inst_late_ratio=e_inst_late,
            inst_avg_payment_ratio=e_inst_pay_ratio,
            inst_underpaid_ratio=e_inst_underpaid,
            cc_avg_utilization=e_cc_util,
            bureau_count=e_bureau_count,
            bureau_overdue_ratio=e_bureau_overdue,
            prev_app_count=e_prev_count,
            prev_refused_ratio=e_prev_refused,
            pos_max_dpd=e_pos_dpd,
        )

        with st.spinner("Running enriched model..."):
            e_result = score_enriched(enriched_input)

        ecol_a, ecol_b, ecol_c = st.columns(3)
        with ecol_a:
            st.metric("Default Probability", f"{e_result.score:.4f}")
        with ecol_b:
            e_icon_map = {"APPROVE": "✅", "REVIEW": "⚠️", "REJECT": "❌"}
            st.metric("Decision", f"{e_icon_map.get(e_result.decision, '')} {e_result.decision}")
        with ecol_c:
            st.metric("Model Version", e_result.model_version)

        e_contrib_data = pd.DataFrame([
            {
                "Feature": fc.feature,
                "Value": round(fc.value, 4),
                "Contribution": round(fc.contribution, 4),
            }
            for fc in e_result.top_features
        ])
        if not e_contrib_data.empty:
            chart_data = e_contrib_data.set_index("Feature")["Contribution"].sort_values()
            st.bar_chart(chart_data, horizontal=True)
            st.dataframe(e_contrib_data, use_container_width=True, hide_index=True)
