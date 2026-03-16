"""Reporting Agent — generates comprehensive reports with graphs and conclusions."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..core.base_agent import BaseAgent
from ..core.messages import AgentMessage
from ..core.types import PipelineStage


class ReportingAgent(BaseAgent):
    """Generates comprehensive Markdown reports covering the full pipeline results."""

    @property
    def stage(self) -> PipelineStage:
        return PipelineStage.REPORTING

    @property
    def description(self) -> str:
        return (
            "Generates a comprehensive report with data analysis summaries, model results, "
            "visualizations, diagnosis findings, and actionable conclusions"
        )

    @property
    def dependencies(self) -> list[PipelineStage]:
        return [PipelineStage.EVALUATION]

    async def execute(self) -> AgentMessage:
        output_dir = Path(self.config.params.get("output_dir", "./output"))
        output_dir.mkdir(parents=True, exist_ok=True)
        report_path = output_dir / "report.md"

        self.logger.info("generating_report")

        sections: list[str] = []

        # Title
        sections.append(self._section_title())

        # Executive summary
        sections.append(self._section_executive_summary())

        # Problem definition
        sections.append(self._section_problem())

        # Data overview
        sections.append(self._section_data_overview())

        # EDA highlights
        sections.append(self._section_eda())

        # Feature engineering
        sections.append(self._section_features())

        # Model selection and training
        sections.append(self._section_modeling())

        # Evaluation results
        sections.append(self._section_evaluation())

        # Diagnosis and optimization
        if self.state.issues or self.state.optimization_history:
            sections.append(self._section_diagnosis())

        # Conclusions and recommendations
        sections.append(self._section_conclusions())

        # Appendix: all plots
        sections.append(self._section_appendix())

        report_content = "\n\n".join(sections)
        report_path.write_text(report_content)

        self.state.report_path = str(report_path)
        self.state.report_content = report_content

        self.logger.info("report_generated", path=str(report_path))

        return self._result_message({
            "report_path": str(report_path),
            "sections": len(sections),
            "length_chars": len(report_content),
        })

    def _section_title(self) -> str:
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        return (
            f"# ML Pipeline Report\n\n"
            f"**Project:** {self.state.problem.description[:100]}\n"
            f"**Generated:** {ts}\n"
            f"**Model:** {self.state.best_model_name}\n"
        )

    def _section_executive_summary(self) -> str:
        metrics = self.state.metrics
        is_acceptable = self.state.is_acceptable
        status = "PASSED" if is_acceptable else "NEEDS IMPROVEMENT"

        lines = [
            "## Executive Summary\n",
            f"**Status:** {status}\n",
            f"**Best Model:** {self.state.best_model_name}\n",
            "**Key Metrics:**\n",
        ]
        for k, v in metrics.items():
            lines.append(f"- {k}: {v:.4f}")

        if not is_acceptable:
            lines.append(f"\n**Issues Found:** {len(self.state.issues)}")
            lines.append(
                f"**Optimization Rounds:** {self.state.optimization_rounds}/{self.state.max_optimization_rounds}"
            )

        return "\n".join(lines)

    def _section_problem(self) -> str:
        p = self.state.problem
        lines = [
            "## Problem Definition\n",
            f"**Type:** {p.problem_type}\n",
            f"**Domain:** {p.domain}\n",
            f"**Target Column:** {p.target_column}\n",
            "**Objectives:**\n",
        ]
        for obj in p.objectives:
            lines.append(f"- {obj}")

        if p.success_criteria:
            lines.append("\n**Success Criteria:**\n")
            for metric, threshold in p.success_criteria.items():
                lines.append(f"- {metric} >= {threshold}")

        return "\n".join(lines)

    def _section_data_overview(self) -> str:
        dp = self.state.data_profile
        lines = [
            "## Data Overview\n",
            f"| Property | Value |",
            f"|----------|-------|",
            f"| Rows | {dp.n_rows:,} |",
            f"| Columns | {dp.n_columns} |",
            f"| Numeric Features | {len(dp.numeric_columns)} |",
            f"| Categorical Features | {len(dp.categorical_columns)} |",
        ]

        missing_cols = {k: v for k, v in dp.missing_counts.items() if v > 0}
        if missing_cols:
            lines.append(f"| Columns with Missing | {len(missing_cols)} |")
            total_missing = sum(missing_cols.values())
            lines.append(f"| Total Missing Values | {total_missing:,} |")

        return "\n".join(lines)

    def _section_eda(self) -> str:
        report = self.state.eda_report
        lines = ["## Exploratory Data Analysis\n"]

        # Missing data
        missing = report.get("missing_data", {})
        if missing:
            lines.append(f"**Missing Data:** {missing.get('missing_percentage', 0)}% of all cells")
            lines.append(f"({missing.get('complete_rows_pct', 0)}% complete rows)\n")

        # Outliers
        outliers = report.get("outliers", {})
        outlier_cols = outliers.get("columns_with_outliers", [])
        if outlier_cols:
            lines.append(f"**Outliers Detected:** {len(outlier_cols)} column(s)\n")
            for col in outlier_cols[:5]:
                detail = outliers.get("details", {}).get(col, {})
                lines.append(
                    f"- {col}: {detail.get('count', 0)} outliers "
                    f"({detail.get('percentage', 0)}%)"
                )

        # Target analysis
        target = report.get("target_analysis", {})
        if target:
            lines.append(f"\n**Target Variable:** {target.get('column', '')}")
            if target.get("type") == "categorical":
                lines.append(f"- Classes: {target.get('n_classes', 0)}")
                lines.append(f"- Balance ratio: {target.get('class_balance_ratio', 0):.4f}")

        # Plots
        for plot_path in self.state.eda_plots:
            lines.append(f"\n![EDA Plot]({plot_path})")

        return "\n".join(lines)

    def _section_features(self) -> str:
        lines = [
            "## Feature Engineering\n",
            f"**Total Features:** {len(self.state.feature_names)}\n",
        ]

        if self.state.encoding_maps:
            lines.append(f"**Encoded Columns:** {len(self.state.encoding_maps)}\n")

        if self.state.X_train is not None:
            lines.append(f"**Training Set Size:** {len(self.state.X_train):,}")
            lines.append(f"**Test Set Size:** {len(self.state.X_test):,}")

        return "\n".join(lines)

    def _section_modeling(self) -> str:
        lines = ["## Model Selection & Training\n"]

        # Selection rationale
        if self.state.selection_rationale:
            lines.append("### Selection Rationale\n")
            lines.append(f"```\n{self.state.selection_rationale}\n```\n")

        # Training results
        history = self.state.training_history
        if history.get("results"):
            lines.append("### Training Results\n")
            lines.append("| Model | CV Mean | CV Std | Time (s) |")
            lines.append("|-------|---------|--------|----------|")
            for r in history["results"]:
                lines.append(
                    f"| {r['name']} | {r['cv_mean']:.4f} | {r['cv_std']:.4f} | {r.get('training_time', 0):.1f} |"
                )

        # Best model hyperparameters
        if self.state.hyperparameters:
            lines.append(f"\n### Best Model: {self.state.best_model_name}\n")
            lines.append("**Tuned Hyperparameters:**\n")
            for k, v in self.state.hyperparameters.items():
                lines.append(f"- {k}: {v}")

        return "\n".join(lines)

    def _section_evaluation(self) -> str:
        lines = ["## Evaluation Results\n"]

        # Metrics table
        metrics = self.state.metrics
        baseline = self.state.baseline_metrics
        lines.append("### Metrics\n")
        lines.append("| Metric | Value |")
        lines.append("|--------|-------|")
        for k, v in metrics.items():
            lines.append(f"| {k} | {v:.4f} |")

        if baseline:
            lines.append("\n### Baseline Comparison\n")
            lines.append("| Baseline Metric | Value |")
            lines.append("|-----------------|-------|")
            for k, v in baseline.items():
                lines.append(f"| {k} | {v:.4f} |")

        # Classification report
        if self.state.classification_report:
            lines.append("\n### Classification Report\n")
            lines.append(f"```\n{self.state.classification_report}\n```")

        # Evaluation plots
        for plot_path in self.state.evaluation_plots:
            lines.append(f"\n![Evaluation Plot]({plot_path})")

        return "\n".join(lines)

    def _section_diagnosis(self) -> str:
        lines = ["## Diagnosis & Optimization\n"]

        if self.state.issues:
            lines.append("### Issues Found\n")
            for issue in self.state.issues:
                severity = issue.get("severity", "INFO")
                desc = issue.get("description", "")
                rec = issue.get("recommendation", "")
                lines.append(f"- **[{severity}]** {desc}")
                if rec:
                    lines.append(f"  - *Recommendation:* {rec}")

        if self.state.optimization_history:
            lines.append("\n### Optimization History\n")
            for opt in self.state.optimization_history:
                lines.append(f"**Round {opt['round']}:** {opt['issues_addressed']} actions taken")
                for action in opt.get("actions", []):
                    lines.append(f"- {action.get('action', '')}")

        return "\n".join(lines)

    def _section_conclusions(self) -> str:
        lines = ["## Conclusions & Recommendations\n"]

        is_acceptable = self.state.is_acceptable
        metrics = self.state.metrics

        if is_acceptable:
            lines.append(
                f"The **{self.state.best_model_name}** model meets the defined success criteria "
                f"and is recommended for deployment consideration.\n"
            )
            lines.append("### Key Findings\n")
            for k, v in metrics.items():
                lines.append(f"- {k}: **{v:.4f}**")

            lines.append("\n### Next Steps\n")
            lines.append("1. Validate on additional holdout data or in A/B test")
            lines.append("2. Monitor for data drift in production")
            lines.append("3. Set up automated retraining pipeline")
            lines.append("4. Document model for stakeholder review")
        else:
            lines.append(
                "The model **does not meet** all success criteria. "
                "Further work is recommended.\n"
            )
            lines.append("### Areas for Improvement\n")
            for issue in self.state.issues:
                if issue.get("severity") in ("error", "critical"):
                    lines.append(f"- {issue.get('description', '')}")

            lines.append("\n### Recommended Actions\n")
            lines.append("1. Collect more training data if possible")
            lines.append("2. Explore additional feature engineering approaches")
            lines.append("3. Try ensemble methods or stacking")
            lines.append("4. Consider domain-specific preprocessing")
            lines.append("5. Review data quality and labeling accuracy")

        return "\n".join(lines)

    def _section_appendix(self) -> str:
        lines = ["## Appendix\n"]

        all_plots = self.state.eda_plots + self.state.evaluation_plots
        if all_plots:
            lines.append("### All Generated Plots\n")
            for i, path in enumerate(all_plots, 1):
                lines.append(f"{i}. `{path}`")

        lines.append(f"\n### Pipeline Stages Completed\n")
        for stage in self.state.completed_stages:
            lines.append(f"- {stage}")

        return "\n".join(lines)
