"""Data utility functions for common DataFrame operations."""

from __future__ import annotations

from typing import Any

import pandas as pd


def detect_column_types(df: pd.DataFrame) -> dict[str, str]:
    """Classify columns into semantic types (numeric, categorical, datetime, text, id)."""
    types: dict[str, str] = {}

    for col in df.columns:
        dtype = df[col].dtype

        if pd.api.types.is_datetime64_any_dtype(dtype):
            types[col] = "datetime"
        elif pd.api.types.is_numeric_dtype(dtype):
            n_unique = df[col].nunique()
            if n_unique <= 2:
                types[col] = "binary"
            elif n_unique <= 20 and n_unique / len(df) < 0.05:
                types[col] = "categorical"
            else:
                types[col] = "numeric"
        elif pd.api.types.is_string_dtype(dtype) or dtype is object:
            avg_len = df[col].dropna().astype(str).str.len().mean()
            n_unique = df[col].nunique()
            if avg_len > 50:
                types[col] = "text"
            elif n_unique == len(df):
                types[col] = "id"
            else:
                types[col] = "categorical"
        else:
            types[col] = "other"

    return types


def compute_missing_pattern(df: pd.DataFrame) -> dict[str, Any]:
    """Analyze missing data patterns across the DataFrame."""
    missing = df.isnull()
    n_rows, n_cols = df.shape

    return {
        "total_missing": int(missing.sum().sum()),
        "total_cells": n_rows * n_cols,
        "missing_rate": float(missing.sum().sum() / (n_rows * n_cols)),
        "complete_rows": int((~missing.any(axis=1)).sum()),
        "complete_rows_pct": float((~missing.any(axis=1)).mean()),
        "columns_with_missing": {
            col: {"count": int(missing[col].sum()), "pct": float(missing[col].mean())}
            for col in df.columns
            if missing[col].any()
        },
    }


def detect_outliers_iqr(
    series: pd.Series, multiplier: float = 1.5
) -> tuple[pd.Series, float, float]:
    """Detect outliers using IQR method. Returns mask, lower_bound, upper_bound."""
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - multiplier * iqr
    upper = q3 + multiplier * iqr
    mask = (series < lower) | (series > upper)
    return mask, float(lower), float(upper)


def safe_value_counts(series: pd.Series, top_n: int = 10) -> dict[str, int]:
    """Get value counts as a JSON-serializable dict."""
    vc = series.value_counts().head(top_n)
    return {str(k): int(v) for k, v in vc.items()}
