"""Data Ingestion Agent — loads and validates data from various sources."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from ..core.base_agent import BaseAgent
from ..core.exceptions import DataValidationError
from ..core.messages import AgentMessage
from ..core.types import PipelineStage
from ..core.workflow_state import DataProfile

# Supported file loaders
_LOADERS: dict[str, Any] = {
    ".csv": pd.read_csv,
    ".tsv": lambda p, **kw: pd.read_csv(p, sep="\t", **kw),
    ".parquet": pd.read_parquet,
    ".json": pd.read_json,
    ".jsonl": lambda p, **kw: pd.read_json(p, lines=True, **kw),
    ".xlsx": pd.read_excel,
    ".xls": pd.read_excel,
    ".feather": pd.read_feather,
}


class DataIngestionAgent(BaseAgent):
    """Loads data from file, validates schema, and computes an initial data profile."""

    @property
    def stage(self) -> PipelineStage:
        return PipelineStage.DATA_INGESTION

    @property
    def description(self) -> str:
        return (
            "Loads data from various file formats, validates schema integrity, "
            "detects column types, and computes an initial data profile"
        )

    @property
    def dependencies(self) -> list[PipelineStage]:
        return [PipelineStage.PROBLEM_ANALYSIS]

    async def execute(self) -> AgentMessage:
        data_source = self.state.data_source
        self.logger.info("ingesting_data", source=data_source)

        # Load the data
        df = self._load_data(data_source)

        # Basic validation
        self._validate(df)

        # Store raw data
        self.state.raw_data = df

        # Compute data profile
        profile = self._compute_profile(df)
        self.state.data_profile = profile

        # Auto-detect target column if not set
        if not self.state.problem.target_column:
            self.state.problem.target_column = self._guess_target_column(df)

        self.logger.info(
            "data_ingested",
            rows=profile.n_rows,
            columns=profile.n_columns,
            target=self.state.problem.target_column,
        )

        return self._result_message(
            {
                "rows": profile.n_rows,
                "columns": profile.n_columns,
                "numeric_columns": len(profile.numeric_columns),
                "categorical_columns": len(profile.categorical_columns),
                "missing_total": sum(profile.missing_counts.values()),
            }
        )

    def _load_data(self, source: str) -> pd.DataFrame:
        """Load data from a file path, auto-detecting format."""
        path = Path(source)
        if not path.exists():
            raise DataValidationError(f"Data file not found: {source}")

        suffix = path.suffix.lower()
        loader = _LOADERS.get(suffix)
        if loader is None:
            raise DataValidationError(
                f"Unsupported file format: {suffix}. Supported: {', '.join(_LOADERS.keys())}"
            )

        try:
            return loader(path)
        except Exception as e:
            raise DataValidationError(f"Failed to load {source}: {e}") from e

    def _validate(self, df: pd.DataFrame) -> None:
        """Basic data validation checks."""
        if df.empty:
            raise DataValidationError("Dataset is empty")

        if len(df.columns) < 2:
            raise DataValidationError("Dataset must have at least 2 columns")

        # Check for duplicate column names
        dupes = df.columns[df.columns.duplicated()].tolist()
        if dupes:
            raise DataValidationError(f"Duplicate column names found: {dupes}")

    def _compute_profile(self, df: pd.DataFrame) -> DataProfile:
        """Compute a data profile with summary statistics."""
        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
        categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        datetime_cols = df.select_dtypes(include=["datetime64"]).columns.tolist()
        text_cols = [c for c in categorical_cols if df[c].dropna().str.len().mean() > 50]
        # Remove text columns from categorical
        categorical_cols = [c for c in categorical_cols if c not in text_cols]

        missing_counts = df.isnull().sum().to_dict()
        missing_pcts = (df.isnull().sum() / len(df) * 100).to_dict()
        unique_counts = df.nunique().to_dict()

        # Summary stats for numeric columns
        summary_stats = {}
        if numeric_cols:
            desc = df[numeric_cols].describe().to_dict()
            summary_stats = {
                col: {str(k): float(v) for k, v in stats.items()} for col, stats in desc.items()
            }

        return DataProfile(
            n_rows=len(df),
            n_columns=len(df.columns),
            column_types={c: str(df[c].dtype) for c in df.columns},
            numeric_columns=numeric_cols,
            categorical_columns=categorical_cols,
            datetime_columns=datetime_cols,
            text_columns=text_cols,
            missing_counts={k: int(v) for k, v in missing_counts.items()},
            missing_percentages={k: round(v, 2) for k, v in missing_pcts.items()},
            unique_counts={k: int(v) for k, v in unique_counts.items()},
            summary_stats=summary_stats,
        )

    def _guess_target_column(self, df: pd.DataFrame) -> str | None:
        """Heuristic to guess the target column if not specified."""
        common_target_names = [
            "target",
            "label",
            "class",
            "y",
            "outcome",
            "response",
            "is_",
            "has_",
            "churned",
            "default",
            "fraud",
        ]
        for col in df.columns:
            col_lower = col.lower()
            for name in common_target_names:
                if name in col_lower:
                    return col

        # Fall back to last column
        return df.columns[-1]
