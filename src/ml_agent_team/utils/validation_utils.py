"""Data validation utilities."""

from __future__ import annotations

from typing import Any

import pandas as pd

from ..core.exceptions import DataValidationError


def validate_dataframe(df: pd.DataFrame, min_rows: int = 1, min_cols: int = 2) -> None:
    """Validate basic DataFrame properties."""
    if df.empty:
        raise DataValidationError("DataFrame is empty")
    if len(df) < min_rows:
        raise DataValidationError(f"DataFrame has {len(df)} rows, minimum is {min_rows}")
    if len(df.columns) < min_cols:
        raise DataValidationError(
            f"DataFrame has {len(df.columns)} columns, minimum is {min_cols}"
        )

    # Check for duplicate columns
    dupes = df.columns[df.columns.duplicated()].tolist()
    if dupes:
        raise DataValidationError(f"Duplicate column names: {dupes}")


def validate_target_column(df: pd.DataFrame, target: str) -> None:
    """Validate that the target column exists and is usable."""
    if target not in df.columns:
        raise DataValidationError(f"Target column '{target}' not found in DataFrame")

    if df[target].isnull().all():
        raise DataValidationError(f"Target column '{target}' is entirely null")

    if df[target].nunique() < 2:
        raise DataValidationError(
            f"Target column '{target}' has fewer than 2 unique values"
        )


def check_data_leakage(
    train_indices: Any, test_indices: Any
) -> bool:
    """Check for overlap between train and test indices."""
    train_set = set(train_indices)
    test_set = set(test_indices)
    overlap = train_set & test_set
    return len(overlap) > 0
