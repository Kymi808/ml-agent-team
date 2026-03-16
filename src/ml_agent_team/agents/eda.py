"""Exploratory Data Analysis Agent — distributions, correlations, outliers, missing patterns."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from ..core.base_agent import BaseAgent
from ..core.messages import AgentMessage
from ..core.types import PipelineStage


class EDAAgent(BaseAgent):
    """Performs extensive exploratory data analysis with statistical summaries and visualizations."""

    @property
    def stage(self) -> PipelineStage:
        return PipelineStage.EDA

    @property
    def description(self) -> str:
        return (
            "Comprehensive exploratory data analysis: distributions, correlations, "
            "outlier detection, missing data patterns, and target analysis"
        )

    @property
    def dependencies(self) -> list[PipelineStage]:
        return [PipelineStage.DATA_INGESTION]

    async def execute(self) -> AgentMessage:
        df = self.state.raw_data
        profile = self.state.data_profile
        target = self.state.problem.target_column
        output_dir = Path(self.config.params.get("output_dir", "./output/eda"))
        output_dir.mkdir(parents=True, exist_ok=True)

        self.logger.info("starting_eda", rows=len(df), columns=len(df.columns))

        report: dict[str, Any] = {}
        plots: list[str] = []

        # 1. Missing data analysis
        report["missing_data"] = self._analyze_missing(df, profile)
        if profile.missing_counts and any(v > 0 for v in profile.missing_counts.values()):
            plot_path = self._plot_missing(df, output_dir)
            if plot_path:
                plots.append(plot_path)

        # 2. Distribution analysis for numeric columns
        report["distributions"] = self._analyze_distributions(df, profile)
        dist_path = self._plot_distributions(df, profile, output_dir)
        if dist_path:
            plots.append(dist_path)

        # 3. Correlation analysis
        report["correlations"] = self._analyze_correlations(df, profile)
        corr_path = self._plot_correlations(df, profile, output_dir)
        if corr_path:
            plots.append(corr_path)
        self.state.correlations = report["correlations"]

        # 4. Outlier detection
        report["outliers"] = self._detect_outliers(df, profile)
        self.state.outlier_report = report["outliers"]

        # 5. Target variable analysis
        if target and target in df.columns:
            report["target_analysis"] = self._analyze_target(df, target, profile)
            target_path = self._plot_target(df, target, output_dir)
            if target_path:
                plots.append(target_path)

        # 6. Categorical analysis
        if profile.categorical_columns:
            report["categorical_analysis"] = self._analyze_categoricals(df, profile)

        self.state.eda_report = report
        self.state.eda_plots = plots

        self.logger.info(
            "eda_complete",
            sections=list(report.keys()),
            plots_generated=len(plots),
        )

        return self._result_message(
            {
                "sections": list(report.keys()),
                "plots_generated": len(plots),
                "outlier_columns": len(report.get("outliers", {}).get("columns_with_outliers", [])),
            }
        )

    def _analyze_missing(self, df: pd.DataFrame, profile: Any) -> dict[str, Any]:
        """Analyze missing data patterns."""
        missing = {col: count for col, count in profile.missing_counts.items() if count > 0}
        total_missing = sum(missing.values())
        total_cells = len(df) * len(df.columns)

        return {
            "total_missing_cells": total_missing,
            "total_cells": total_cells,
            "missing_percentage": round(total_missing / total_cells * 100, 2) if total_cells else 0,
            "columns_with_missing": missing,
            "complete_rows": int(df.dropna().shape[0]),
            "complete_rows_pct": round(df.dropna().shape[0] / len(df) * 100, 2),
        }

    def _analyze_distributions(self, df: pd.DataFrame, profile: Any) -> dict[str, Any]:
        """Compute distribution stats for numeric columns."""
        results = {}
        for col in profile.numeric_columns:
            series = df[col].dropna()
            if len(series) == 0:
                continue
            results[col] = {
                "mean": float(series.mean()),
                "median": float(series.median()),
                "std": float(series.std()),
                "skewness": float(series.skew()),
                "kurtosis": float(series.kurtosis()),
                "min": float(series.min()),
                "max": float(series.max()),
                "q25": float(series.quantile(0.25)),
                "q75": float(series.quantile(0.75)),
            }
        return results

    def _analyze_correlations(self, df: pd.DataFrame, profile: Any) -> dict[str, dict[str, float]]:
        """Compute correlation matrix for numeric columns."""
        if len(profile.numeric_columns) < 2:
            return {}
        corr = df[profile.numeric_columns].corr(method="pearson")
        return {
            col: {c: round(float(v), 4) for c, v in row.items()}
            for col, row in corr.to_dict().items()
        }

    def _detect_outliers(self, df: pd.DataFrame, profile: Any) -> dict[str, Any]:
        """Detect outliers using the IQR method."""
        outlier_info: dict[str, Any] = {"columns_with_outliers": [], "details": {}}

        for col in profile.numeric_columns:
            series = df[col].dropna()
            if len(series) == 0:
                continue
            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            outlier_mask = (series < lower) | (series > upper)
            n_outliers = int(outlier_mask.sum())

            if n_outliers > 0:
                outlier_info["columns_with_outliers"].append(col)
                outlier_info["details"][col] = {
                    "count": n_outliers,
                    "percentage": round(n_outliers / len(series) * 100, 2),
                    "lower_bound": float(lower),
                    "upper_bound": float(upper),
                }

        return outlier_info

    def _analyze_target(self, df: pd.DataFrame, target: str, profile: Any) -> dict[str, Any]:
        """Analyze the target variable."""
        series = df[target].dropna()
        result: dict[str, Any] = {"column": target, "dtype": str(series.dtype)}

        if target in profile.numeric_columns:
            result["type"] = "continuous"
            result["mean"] = float(series.mean())
            result["std"] = float(series.std())
            result["min"] = float(series.min())
            result["max"] = float(series.max())
        else:
            result["type"] = "categorical"
            vc = series.value_counts()
            result["class_distribution"] = {str(k): int(v) for k, v in vc.items()}
            result["n_classes"] = int(vc.shape[0])
            result["class_balance_ratio"] = round(float(vc.min() / vc.max()), 4)

        return result

    def _analyze_categoricals(self, df: pd.DataFrame, profile: Any) -> dict[str, Any]:
        """Analyze categorical columns."""
        results = {}
        for col in profile.categorical_columns:
            vc = df[col].value_counts()
            results[col] = {
                "n_unique": int(vc.shape[0]),
                "top_values": {str(k): int(v) for k, v in vc.head(10).items()},
                "mode": str(vc.index[0]) if len(vc) > 0 else None,
            }
        return results

    # ── Plotting methods ──

    def _plot_missing(self, df: pd.DataFrame, output_dir: Path) -> str | None:
        """Plot missing data heatmap."""
        missing_cols = [c for c in df.columns if df[c].isnull().any()]
        if not missing_cols:
            return None

        fig, ax = plt.subplots(figsize=(12, max(4, len(missing_cols) * 0.3)))
        missing_pct = df[missing_cols].isnull().mean().sort_values(ascending=True)
        missing_pct.plot.barh(ax=ax, color="coral")
        ax.set_xlabel("Missing Fraction")
        ax.set_title("Missing Data by Column")
        plt.tight_layout()

        path = str(output_dir / "missing_data.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        return path

    def _plot_distributions(self, df: pd.DataFrame, profile: Any, output_dir: Path) -> str | None:
        """Plot histograms for numeric columns."""
        cols = profile.numeric_columns[:20]  # Limit to 20 columns
        if not cols:
            return None

        n_cols = min(4, len(cols))
        n_rows = (len(cols) + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows))
        axes_flat = np.array(axes).flatten() if len(cols) > 1 else [axes]

        for i, col in enumerate(cols):
            ax = axes_flat[i]
            df[col].dropna().hist(bins=30, ax=ax, color="steelblue", edgecolor="white")
            ax.set_title(col, fontsize=10)
            ax.tick_params(labelsize=8)

        # Hide unused axes
        for j in range(len(cols), len(axes_flat)):
            axes_flat[j].set_visible(False)

        fig.suptitle("Feature Distributions", fontsize=14)
        plt.tight_layout()

        path = str(output_dir / "distributions.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        return path

    def _plot_correlations(self, df: pd.DataFrame, profile: Any, output_dir: Path) -> str | None:
        """Plot correlation heatmap."""
        cols = profile.numeric_columns[:20]
        if len(cols) < 2:
            return None

        corr = df[cols].corr()
        fig, ax = plt.subplots(figsize=(max(8, len(cols) * 0.6), max(6, len(cols) * 0.5)))
        sns.heatmap(
            corr,
            annot=len(cols) <= 12,
            fmt=".2f",
            cmap="RdBu_r",
            center=0,
            ax=ax,
            square=True,
            linewidths=0.5,
        )
        ax.set_title("Feature Correlations")
        plt.tight_layout()

        path = str(output_dir / "correlations.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        return path

    def _plot_target(self, df: pd.DataFrame, target: str, output_dir: Path) -> str | None:
        """Plot target variable distribution."""
        if target not in df.columns:
            return None

        fig, ax = plt.subplots(figsize=(8, 5))
        series = df[target].dropna()

        if pd.api.types.is_numeric_dtype(series) and series.nunique() > 10:
            series.hist(bins=30, ax=ax, color="steelblue", edgecolor="white")
            ax.set_xlabel(target)
            ax.set_ylabel("Count")
        else:
            vc = series.value_counts()
            vc.plot.bar(ax=ax, color="steelblue", edgecolor="white")
            ax.set_ylabel("Count")

        ax.set_title(f"Target Variable: {target}")
        plt.tight_layout()

        path = str(output_dir / "target_distribution.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        return path
