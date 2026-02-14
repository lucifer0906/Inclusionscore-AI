"""Training orchestration (data preparation, model fitting, saving)."""

from __future__ import annotations

import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

from src.config import MODEL_DIR, MODEL_NAME, RANDOM_SEED
from src.data_loader import load_primary_dataset
from src.features import TARGET_COL, clean_df, create_features, save_processed
from src.models import (
    evaluate, find_optimal_threshold, save_model, train_baseline, train_xgboost,
)
from src.tuning import load_best_params


def prepare_data(
    test_size: float = 0.15,
    val_size: float = 0.15,
) -> tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame,
    pd.Series, pd.Series, pd.Series,
]:
    """Load raw data, clean, engineer features, and produce stratified splits.

    Split strategy (default sizes applied to 150 k rows):
        1. 70 % train, 15 % validation, 15 % test
        2. Stratified on ``SeriousDlqin2yrs`` to preserve class ratio

    Parameters
    ----------
    test_size : float
        Fraction of data reserved for the test set.
    val_size : float
        Fraction of the *remaining* data (after test split) used for
        validation.  With defaults the effective sizes are
        ~70 / 15 / 15 %.

    Returns
    -------
    tuple
        ``(X_train, X_val, X_test, y_train, y_val, y_test)``
    """
    # 1. Load
    df_raw = load_primary_dataset()

    # 2. Clean
    df_clean = clean_df(df_raw, fit=True)

    # 3. Feature engineering
    df_feat = create_features(df_clean)

    # 4. Separate target
    y = df_feat[TARGET_COL]
    X = df_feat.drop(columns=[TARGET_COL])

    # 5. Stratified train / temp split
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y,
        test_size=test_size + val_size,
        random_state=RANDOM_SEED,
        stratify=y,
    )

    # 6. Split temp into val / test (equal halves of the temp set)
    relative_test = test_size / (test_size + val_size)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=relative_test,
        random_state=RANDOM_SEED,
        stratify=y_temp,
    )

    # 7. Reset indices
    for frame in (X_train, X_val, X_test):
        frame.reset_index(drop=True, inplace=True)
    for series in (y_train, y_val, y_test):
        series.reset_index(drop=True, inplace=True)

    # 8. Persist
    save_processed(X_train, X_val, X_test, y_train, y_val, y_test)

    print(f"Train : {X_train.shape[0]:,} rows  (default rate {y_train.mean():.3f})")
    print(f"Val   : {X_val.shape[0]:,} rows  (default rate {y_val.mean():.3f})")
    print(f"Test  : {X_test.shape[0]:,} rows  (default rate {y_test.mean():.3f})")

    return X_train, X_val, X_test, y_train, y_val, y_test


def run_pipeline() -> None:
    """Execute the full training pipeline: data prep → train → evaluate → save."""
    import src.features as _feat_mod

    X_train, X_val, X_test, y_train, y_val, y_test = prepare_data()

    # Read AFTER prepare_data() so clean_df(fit=True) has set the value.
    # Using module attribute access (not `from ... import`) because
    # `from` would bind the initial None before clean_df runs.
    debt_ratio_cap = _feat_mod.last_debt_ratio_cap
    income_fill = _feat_mod.last_income_fill
    dep_fill = _feat_mod.last_dep_fill

    # 0. Apply SMOTE to training data only (preserve val/test distributions)
    print("\n-- SMOTE oversampling (training set only) --")
    print(f"  Before: {len(X_train):,} rows  "
          f"(class 0: {(y_train == 0).sum():,}, class 1: {(y_train == 1).sum():,})")
    smote = SMOTE(random_state=RANDOM_SEED)
    X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)
    X_train_sm = pd.DataFrame(X_train_sm, columns=X_train.columns)
    y_train_sm = pd.Series(y_train_sm, name=y_train.name)
    print(f"  After:  {len(X_train_sm):,} rows  "
          f"(class 0: {(y_train_sm == 0).sum():,}, class 1: {(y_train_sm == 1).sum():,})")

    # Use SMOTE'd data for training, original data for evaluation
    X_train = X_train_sm
    y_train = y_train_sm

    # 1. Logistic Regression baseline
    print("\n-- Logistic Regression baseline --")
    lr_model, lr_metrics = train_baseline(X_train, y_train)
    lr_test_metrics = evaluate(lr_model, X_test, y_test, label="LR-test")
    print(f"  Val  AUC: {lr_metrics['auc_roc']:.4f}")
    print(f"  Test AUC: {lr_test_metrics['auc_roc']:.4f}")

    # 2. XGBoost (use tuned params if available)
    print("\n-- XGBoost --")
    best_params = load_best_params()
    if best_params:
        print(f"  Using tuned params from models/best_params.json")
        xgb_config = best_params
    else:
        print(f"  Using default params (run src/tuning.py for optimization)")
        xgb_config = None

    # SMOTE already balanced the classes (1:1), so the Optuna-tuned
    # scale_pos_weight (~11.4, calibrated for the original ~1:14 ratio)
    # would over-compensate.  Reset to 1.0 so both strategies contribute
    # without interference: SMOTE handles data augmentation, while
    # scale_pos_weight remains available for fine-tuning if needed.
    if xgb_config and "scale_pos_weight" in xgb_config:
        original_spw = xgb_config["scale_pos_weight"]
        xgb_config = {**xgb_config, "scale_pos_weight": 1.0}
        print(f"  Adjusted scale_pos_weight: {original_spw:.2f} -> 1.0 (post-SMOTE)")

    xgb_model, xgb_metrics = train_xgboost(
        X_train, y_train, X_val, y_val,
        config=xgb_config,
    )
    xgb_test_metrics = evaluate(xgb_model, X_test, y_test, label="XGB-test")
    print(f"  Val  AUC: {xgb_metrics['auc_roc']:.4f}")
    print(f"  Test AUC: {xgb_test_metrics['auc_roc']:.4f}")

    # 3. Find optimal decision threshold on validation set
    print("\n-- Threshold optimization --")
    thresh_info = find_optimal_threshold(xgb_model, X_val, y_val)
    print(f"  Optimal threshold: {thresh_info['optimal_threshold']:.4f}")
    print(f"  F1 at threshold:   {thresh_info['f1_at_threshold']:.4f}")
    print(f"  Precision:         {thresh_info['precision_at_threshold']:.4f}")
    print(f"  Recall:            {thresh_info['recall_at_threshold']:.4f}")

    # 4. Save best model (XGBoost) with preprocessing metadata
    path = save_model(
        xgb_model,
        xgb_test_metrics,
        name=MODEL_NAME,
        output_dir=MODEL_DIR,
        X_train_rows=len(X_train),
        X_val_rows=len(X_val),
        extra_meta={
            "preprocessing": {
                "debt_ratio_cap": debt_ratio_cap,
                "income_fill": income_fill,
                "dep_fill": dep_fill,
                "smote": True,
                "smote_strategy": "auto",
            },
            "threshold_optimization": thresh_info,
        },
    )
    print(f"\nModel saved -> {path}")


if __name__ == "__main__":
    run_pipeline()
