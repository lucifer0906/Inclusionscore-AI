"""
Data loading and inspection utilities.

Provides helpers to list available dataset files, inspect CSV headers,
and load the primary dataset for modelling.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import pandas as pd

from src.config import DATA_DIR, PRIMARY_DATASET_SUBDIR, PRIMARY_TRAIN_FILE


# ── Discovery helpers ────────────────────────────────────────────────────

def list_data_files(root: Optional[Path] = None) -> dict[str, list[str]]:
    """Return ``{subfolder: [file_names]}`` for every subfolder in *root*.

    Parameters
    ----------
    root : Path, optional
        Top-level data directory. Defaults to ``DATA_DIR`` from config.

    Returns
    -------
    dict[str, list[str]]
        Mapping of subfolder name → sorted list of file names.
    """
    root = root or DATA_DIR
    inventory: dict[str, list[str]] = {}
    for entry in sorted(root.iterdir()):
        if entry.is_dir():
            files = sorted(f.name for f in entry.iterdir() if f.is_file())
            inventory[entry.name] = files
    return inventory


def inspect_file_headers(
    path: str | Path,
    n_rows: int = 2,
) -> tuple[list[str], pd.DataFrame]:
    """Read the first *n_rows* of a CSV (or Excel) file and return headers + sample.

    Parameters
    ----------
    path : str | Path
        Full path to the file.
    n_rows : int
        Number of sample rows to return (default 2).

    Returns
    -------
    tuple[list[str], pd.DataFrame]
        Column names and a small preview DataFrame.
    """
    path = Path(path)
    if path.suffix in (".xlsx", ".xls"):
        df = pd.read_excel(path, nrows=n_rows)
    else:
        df = pd.read_csv(path, nrows=n_rows)
    return list(df.columns), df


# ── Loading helpers ──────────────────────────────────────────────────────

def load_csv(
    path: str | Path,
    **kwargs,
) -> pd.DataFrame:
    """Safe CSV loader with sensible defaults.

    Parameters
    ----------
    path : str | Path
        Path to the CSV file.
    **kwargs
        Forwarded to ``pd.read_csv``.

    Returns
    -------
    pd.DataFrame
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    return pd.read_csv(path, **kwargs)


def load_primary_dataset() -> pd.DataFrame:
    """Load the primary *Give Me Some Credit* training CSV.

    Returns
    -------
    pd.DataFrame
        Raw training data.
    """
    path = DATA_DIR / PRIMARY_DATASET_SUBDIR / PRIMARY_TRAIN_FILE
    df = load_csv(path, index_col=0)
    return df


# ── CLI / __main__ ───────────────────────────────────────────────────────

def _print_inventory() -> str:
    """Print a human-readable data inventory and return it as text."""
    lines: list[str] = []
    lines.append("=" * 60)
    lines.append("DATA INVENTORY")
    lines.append("=" * 60)

    inventory = list_data_files()
    if not inventory:
        lines.append("(no subfolders found in data directory)")
        text = "\n".join(lines)
        print(text)
        return text

    for folder, files in inventory.items():
        lines.append(f"\n-- {folder}/ ({len(files)} files) --")
        for fname in files:
            lines.append(f"   * {fname}")

        # Inspect headers for CSV / Excel files
        for fname in files:
            fpath = DATA_DIR / folder / fname
            if fpath.suffix in (".csv", ".xlsx", ".xls"):
                try:
                    cols, sample = inspect_file_headers(fpath)
                    lines.append(f"\n   [{fname}] columns ({len(cols)}):")
                    lines.append(f"   {cols}")
                    lines.append(f"   First 2 rows:")
                    for row in sample.to_string(index=False).split("\n"):
                        lines.append(f"      {row}")
                except Exception as exc:
                    lines.append(f"   [{fname}] WARNING could not read: {exc}")

    lines.append("\n" + "=" * 60)
    text = "\n".join(lines)
    print(text)
    return text


if __name__ == "__main__":
    _print_inventory()
