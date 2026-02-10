"""General-purpose utility functions."""

from __future__ import annotations

from typing import Any

import pandas as pd


def summarize_df(df: pd.DataFrame) -> dict[str, Any]:
    """Return a JSON-serialisable summary of *df*.

    Includes row/column counts, dtype breakdown, null counts, and basic
    descriptive statistics for numeric columns.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.

    Returns
    -------
    dict[str, Any]
        Summary dictionary with keys: ``rows``, ``cols``, ``columns``,
        ``dtypes``, ``nulls``, ``null_pct``, ``describe``.
    """
    null_counts = df.isnull().sum()
    summary: dict[str, Any] = {
        "rows": int(df.shape[0]),
        "cols": int(df.shape[1]),
        "columns": list(df.columns),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "nulls": {col: int(v) for col, v in null_counts.items()},
        "null_pct": {
            col: round(v / len(df) * 100, 2)
            for col, v in null_counts.items()
        },
        "describe": df.describe().to_dict(),
    }
    return summary
