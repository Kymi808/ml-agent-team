"""File I/O utility functions."""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import pandas as pd


def load_dataframe(path: str | Path, **kwargs: Any) -> pd.DataFrame:
    """Load a DataFrame from file, auto-detecting format."""
    path = Path(path)
    suffix = path.suffix.lower()

    loaders = {
        ".csv": pd.read_csv,
        ".tsv": lambda p, **kw: pd.read_csv(p, sep="\t", **kw),
        ".parquet": pd.read_parquet,
        ".json": pd.read_json,
        ".jsonl": lambda p, **kw: pd.read_json(p, lines=True, **kw),
        ".xlsx": pd.read_excel,
        ".xls": pd.read_excel,
        ".feather": pd.read_feather,
    }

    loader = loaders.get(suffix)
    if loader is None:
        raise ValueError(f"Unsupported file format: {suffix}")

    return loader(path, **kwargs)


def save_dataframe(df: pd.DataFrame, path: str | Path, **kwargs: Any) -> None:
    """Save a DataFrame to file, auto-detecting format."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()

    if suffix == ".csv":
        df.to_csv(path, index=False, **kwargs)
    elif suffix == ".parquet":
        df.to_parquet(path, index=False, **kwargs)
    elif suffix == ".json":
        df.to_json(path, orient="records", **kwargs)
    else:
        df.to_csv(path, index=False, **kwargs)


def save_pickle(obj: Any, path: str | Path) -> None:
    """Save an object using pickle."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(path: str | Path) -> Any:
    """Load a pickled object."""
    with open(path, "rb") as f:
        return pickle.load(f)  # noqa: S301


def save_json(data: Any, path: str | Path) -> None:
    """Save data as JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)


def load_json(path: str | Path) -> Any:
    """Load JSON data."""
    with open(path) as f:
        return json.load(f)
