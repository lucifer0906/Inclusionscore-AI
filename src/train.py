"""Training orchestration (data preparation, model fitting, saving)."""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import RANDOM_SEED
from src.data_loader import load_primary_dataset
from src.features import TARGET_COL, clean_df, create_features, save_processed


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


if __name__ == "__main__":
    prepare_data()
