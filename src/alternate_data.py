"""Alternate data feature engineering from Home Credit Default Risk dataset.

Demonstrates how transaction patterns, payment behaviour, and financial
activity from non-traditional sources improve credit scoring for
unbanked / under-banked populations.

Data tables used (all from Kaggle Home Credit Default Risk):
- installments_payments.csv  -- loan instalment payment history
- credit_card_balance.csv    -- credit card monthly snapshots
- POS_CASH_balance.csv       -- point-of-sale / cash loan balances
- previous_application.csv   -- historical loan applications
- bureau.csv                 -- external credit bureau records

Each table is aggregated per applicant (SK_ID_CURR) into a fixed-width
feature vector that can be joined to the main application table.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.config import DATA_DIR

HOME_CREDIT_DIR = DATA_DIR / "home-credit-default-risk"


# -- Installment payment behaviour -------------------------------------------

def _engineer_installment_features(path: Path | None = None) -> pd.DataFrame:
    """Engineer features from instalment payment records.

    Captures payment timeliness, payment completion ratio, and
    behavioural consistency -- proxies for financial discipline even
    when no formal credit score exists.
    """
    path = path or HOME_CREDIT_DIR / "installments_payments.csv"
    df = pd.read_csv(path)

    # Days difference: negative = paid early, positive = paid late
    df["payment_delay"] = df["DAYS_ENTRY_PAYMENT"] - df["DAYS_INSTALMENT"]
    df["is_late"] = (df["payment_delay"] > 0).astype(int)
    df["is_early"] = (df["payment_delay"] < -5).astype(int)

    # Payment ratio: 1.0 = paid in full, <1 = under-paid
    df["payment_ratio"] = (
        df["AMT_PAYMENT"] / df["AMT_INSTALMENT"].replace(0, np.nan)
    )
    df["is_underpaid"] = (df["payment_ratio"] < 0.95).astype(int)

    agg = df.groupby("SK_ID_CURR").agg(
        inst_count=("payment_delay", "count"),
        inst_avg_delay=("payment_delay", "mean"),
        inst_max_delay=("payment_delay", "max"),
        inst_late_ratio=("is_late", "mean"),
        inst_early_ratio=("is_early", "mean"),
        inst_avg_payment_ratio=("payment_ratio", "mean"),
        inst_underpaid_ratio=("is_underpaid", "mean"),
    ).reset_index()

    return agg


# -- Credit card usage patterns ----------------------------------------------

def _engineer_credit_card_features(path: Path | None = None) -> pd.DataFrame:
    """Engineer features from credit card monthly balance snapshots.

    Captures utilisation patterns, drawing behaviour (ATM vs POS), and
    days-past-due trends -- signals of financial stress or stability.
    """
    path = path or HOME_CREDIT_DIR / "credit_card_balance.csv"
    df = pd.read_csv(path)

    df["cc_utilization"] = (
        df["AMT_BALANCE"] / df["AMT_CREDIT_LIMIT_ACTUAL"].replace(0, np.nan)
    )

    agg = df.groupby("SK_ID_CURR").agg(
        cc_months=("MONTHS_BALANCE", "count"),
        cc_avg_utilization=("cc_utilization", "mean"),
        cc_max_utilization=("cc_utilization", "max"),
        cc_avg_balance=("AMT_BALANCE", "mean"),
        cc_avg_drawings=("AMT_DRAWINGS_CURRENT", "mean"),
        cc_avg_atm_drawings=("AMT_DRAWINGS_ATM_CURRENT", "mean"),
        cc_avg_payment=("AMT_PAYMENT_CURRENT", "mean"),
        cc_max_dpd=("SK_DPD", "max"),
        cc_avg_dpd=("SK_DPD", "mean"),
    ).reset_index()

    return agg


# -- POS / cash loan transaction behaviour -----------------------------------

def _engineer_pos_features(path: Path | None = None) -> pd.DataFrame:
    """Engineer features from point-of-sale and cash loan balances.

    POS (point-of-sale) loans are instalment purchases at retail stores.
    These capture the applicant's real-world transaction footprint --
    analogous to UPI / e-commerce purchase patterns.
    """
    path = path or HOME_CREDIT_DIR / "POS_CASH_balance.csv"
    df = pd.read_csv(path)

    df["is_completed"] = (
        df["NAME_CONTRACT_STATUS"] == "Completed"
    ).astype(int)

    agg = df.groupby("SK_ID_CURR").agg(
        pos_months=("MONTHS_BALANCE", "count"),
        pos_completed_ratio=("is_completed", "mean"),
        pos_max_dpd=("SK_DPD", "max"),
        pos_avg_dpd=("SK_DPD", "mean"),
        pos_avg_instalments_left=("CNT_INSTALMENT_FUTURE", "mean"),
    ).reset_index()

    return agg


# -- Previous loan application history ---------------------------------------

def _engineer_prev_app_features(path: Path | None = None) -> pd.DataFrame:
    """Engineer features from previous loan applications.

    The number of prior applications and their outcomes (approved vs
    refused) reveal the applicant's borrowing history -- a form of
    alternative data when traditional credit bureau data is thin.
    """
    path = path or HOME_CREDIT_DIR / "previous_application.csv"
    df = pd.read_csv(path)

    df["was_approved"] = (
        df["NAME_CONTRACT_STATUS"] == "Approved"
    ).astype(int)
    df["was_refused"] = (
        df["NAME_CONTRACT_STATUS"] == "Refused"
    ).astype(int)

    agg = df.groupby("SK_ID_CURR").agg(
        prev_app_count=("SK_ID_PREV", "nunique"),
        prev_approved_ratio=("was_approved", "mean"),
        prev_refused_ratio=("was_refused", "mean"),
        prev_avg_credit=("AMT_CREDIT", "mean"),
        prev_avg_annuity=("AMT_ANNUITY", "mean"),
    ).reset_index()

    return agg


# -- Bureau (external credit records) ----------------------------------------

def _engineer_bureau_features(path: Path | None = None) -> pd.DataFrame:
    """Engineer features from credit bureau records.

    Even for under-banked applicants, partial bureau data (e.g.,
    microfinance records, utility-linked credit) may exist. These
    features capture whatever history is available.
    """
    path = path or HOME_CREDIT_DIR / "bureau.csv"
    df = pd.read_csv(path)

    df["is_active"] = (df["CREDIT_ACTIVE"] == "Active").astype(int)
    df["has_overdue"] = (df["CREDIT_DAY_OVERDUE"] > 0).astype(int)

    agg = df.groupby("SK_ID_CURR").agg(
        bureau_count=("SK_ID_BUREAU", "nunique"),
        bureau_active_ratio=("is_active", "mean"),
        bureau_overdue_ratio=("has_overdue", "mean"),
        bureau_avg_credit=("AMT_CREDIT_SUM", "mean"),
        bureau_max_overdue_amt=("AMT_CREDIT_MAX_OVERDUE", "max"),
        bureau_avg_debt=("AMT_CREDIT_SUM_DEBT", "mean"),
    ).reset_index()

    return agg


# -- Public API ---------------------------------------------------------------

def load_alternate_features() -> pd.DataFrame:
    """Load and merge all alternate data features into one DataFrame.

    Returns a DataFrame indexed by SK_ID_CURR with ~30 engineered
    features from five alternate data tables.
    """
    print("Engineering alternate data features...")

    print("  [1/5] Installment payment behaviour")
    inst = _engineer_installment_features()

    print("  [2/5] Credit card usage patterns")
    cc = _engineer_credit_card_features()

    print("  [3/5] POS / cash transaction behaviour")
    pos = _engineer_pos_features()

    print("  [4/5] Previous application history")
    prev = _engineer_prev_app_features()

    print("  [5/5] Bureau credit records")
    bur = _engineer_bureau_features()

    # Merge all on SK_ID_CURR (outer join to keep all applicants)
    merged = inst
    for right in [cc, pos, prev, bur]:
        merged = merged.merge(right, on="SK_ID_CURR", how="outer")

    print(f"  Total: {len(merged):,} applicants, "
          f"{len(merged.columns) - 1} alternate features")

    return merged


def build_enriched_dataset() -> tuple[pd.DataFrame, pd.Series]:
    """Build a feature matrix combining application data with alternate features.

    Returns (X, y) where X includes both traditional application
    features and alternate data features from the Home Credit tables.
    """
    app_path = HOME_CREDIT_DIR / "application_train.csv"
    app = pd.read_csv(app_path)

    y = app["TARGET"]
    # Select a focused set of numeric application features
    app_features = [
        "SK_ID_CURR",
        "AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY", "AMT_GOODS_PRICE",
        "DAYS_BIRTH", "DAYS_EMPLOYED", "DAYS_REGISTRATION", "DAYS_ID_PUBLISH",
        "EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3",
        "CNT_CHILDREN", "CNT_FAM_MEMBERS",
        "REGION_RATING_CLIENT",
        "DAYS_LAST_PHONE_CHANGE",
        "OBS_30_CNT_SOCIAL_CIRCLE", "DEF_30_CNT_SOCIAL_CIRCLE",
    ]
    X_app = app[app_features].copy()

    # Load alternate data features
    alt = load_alternate_features()

    # Merge
    X = X_app.merge(alt, on="SK_ID_CURR", how="left")
    X = X.drop(columns=["SK_ID_CURR"])

    # Fill NaN (applicants with no alternate data records)
    X = X.fillna(0)

    print(f"\nEnriched dataset: {X.shape[0]:,} rows x {X.shape[1]} features")
    print(f"  Application features: {len(app_features) - 1}")
    print(f"  Alternate data features: {X.shape[1] - len(app_features) + 1}")

    return X, y


def run_alternate_data_demo() -> dict:
    """Demonstrate that alternate data features improve default prediction.

    Trains two XGBoost models on the Home Credit dataset:
    1. Application-only features (traditional)
    2. Application + alternate data features (enriched)

    Returns a dict with both models' AUC-ROC scores.
    """
    from sklearn.model_selection import train_test_split
    from xgboost import XGBClassifier
    from sklearn.metrics import roc_auc_score

    from src.config import RANDOM_SEED

    print("=" * 60)
    print("  ALTERNATE DATA DEMO")
    print("  Comparing traditional vs. enriched feature sets")
    print("=" * 60)

    # Load enriched dataset
    X, y = build_enriched_dataset()

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y,
    )

    # Identify feature groups
    app_cols = [
        "AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY", "AMT_GOODS_PRICE",
        "DAYS_BIRTH", "DAYS_EMPLOYED", "DAYS_REGISTRATION", "DAYS_ID_PUBLISH",
        "EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3",
        "CNT_CHILDREN", "CNT_FAM_MEMBERS",
        "REGION_RATING_CLIENT",
        "DAYS_LAST_PHONE_CHANGE",
        "OBS_30_CNT_SOCIAL_CIRCLE", "DEF_30_CNT_SOCIAL_CIRCLE",
    ]
    alt_cols = [c for c in X.columns if c not in app_cols]

    # Model 1: Application-only
    print("\n-- Model 1: Application features only --")
    xgb1 = XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        eval_metric="auc", random_state=RANDOM_SEED, n_jobs=-1,
    )
    xgb1.fit(X_train[app_cols], y_train,
             eval_set=[(X_test[app_cols], y_test)], verbose=0)
    auc_app = roc_auc_score(y_test, xgb1.predict_proba(X_test[app_cols])[:, 1])
    print(f"  AUC-ROC (application only): {auc_app:.4f}")

    # Model 2: Application + alternate data
    print("\n-- Model 2: Application + alternate data features --")
    xgb2 = XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        eval_metric="auc", random_state=RANDOM_SEED, n_jobs=-1,
    )
    xgb2.fit(X_train, y_train,
             eval_set=[(X_test, y_test)], verbose=0)
    auc_full = roc_auc_score(y_test, xgb2.predict_proba(X_test)[:, 1])
    print(f"  AUC-ROC (app + alternate):  {auc_full:.4f}")

    improvement = auc_full - auc_app
    print(f"\n  Improvement from alternate data: +{improvement:.4f} AUC")
    print(f"  Alternate features used: {len(alt_cols)}")

    # Top alternate data features by importance
    importances = pd.Series(
        xgb2.feature_importances_, index=X.columns,
    ).sort_values(ascending=False)
    alt_importances = importances[importances.index.isin(alt_cols)].head(10)
    print("\n  Top 10 alternate data features by importance:")
    for feat, imp in alt_importances.items():
        print(f"    {feat:35s}  {imp:.4f}")

    results = {
        "auc_application_only": round(auc_app, 4),
        "auc_with_alternate_data": round(auc_full, 4),
        "improvement": round(improvement, 4),
        "n_app_features": len(app_cols),
        "n_alt_features": len(alt_cols),
        "top_alt_features": list(alt_importances.head(5).index),
    }

    return results


if __name__ == "__main__":
    run_alternate_data_demo()
