"""Peer Review Agent — validates methodology, catches data leakage, and ensures quality."""

from __future__ import annotations

from typing import Any

import numpy as np

from ..core.base_agent import BaseAgent
from ..core.messages import AgentMessage
from ..core.types import PipelineStage, Severity


class PeerReviewAgent(BaseAgent):
    """Cross-cutting agent that reviews other agents' work for methodology issues.

    Validates:
    - Data leakage (target leaking into features)
    - Statistical validity (sufficient samples, meaningful results)
    - Reproducibility (random seeds, deterministic operations)
    - Best practices (proper cross-validation, appropriate metrics)
    """

    @property
    def stage(self) -> PipelineStage:
        return PipelineStage.REPORTING  # Runs alongside reporting

    @property
    def description(self) -> str:
        return (
            "Reviews the work of all pipeline agents for methodology issues, "
            "data leakage, statistical validity, and reproducibility"
        )

    @property
    def dependencies(self) -> list[PipelineStage]:
        return [PipelineStage.EVALUATION]

    async def execute(self) -> AgentMessage:
        self.logger.info("starting_peer_review")

        findings: list[dict[str, Any]] = []

        # Review data handling
        findings.extend(self._review_data_handling())

        # Review feature engineering
        findings.extend(self._review_feature_engineering())

        # Review model training
        findings.extend(self._review_training())

        # Review evaluation
        findings.extend(self._review_evaluation())

        # Review overall methodology
        findings.extend(self._review_methodology())

        # Determine approval
        critical = [f for f in findings if f["severity"] == Severity.CRITICAL]
        errors = [f for f in findings if f["severity"] == Severity.ERROR]
        approved = len(critical) == 0

        self.state.review_findings = findings
        self.state.review_approved = approved

        self.logger.info(
            "peer_review_complete",
            total_findings=len(findings),
            critical=len(critical),
            errors=len(errors),
            approved=approved,
        )

        return self._result_message({
            "approved": approved,
            "total_findings": len(findings),
            "critical": len(critical),
            "errors": len(errors),
            "findings": [
                {"severity": f["severity"], "category": f["category"], "message": f["message"]}
                for f in findings
            ],
        })

    def _review_data_handling(self) -> list[dict[str, Any]]:
        """Review data ingestion and preprocessing."""
        findings: list[dict[str, Any]] = []
        profile = self.state.data_profile

        # Check for very small dataset
        if profile.n_rows < 100:
            findings.append({
                "severity": Severity.WARNING,
                "category": "data",
                "message": (
                    f"Very small dataset ({profile.n_rows} rows). "
                    f"Results may not be statistically reliable."
                ),
                "recommendation": "Collect more data or use leave-one-out cross-validation",
            })

        # Check for high missing rate
        total_cells = profile.n_rows * profile.n_columns
        total_missing = sum(profile.missing_counts.values())
        if total_cells > 0 and total_missing / total_cells > 0.3:
            findings.append({
                "severity": Severity.WARNING,
                "category": "data",
                "message": (
                    f"High missing data rate ({total_missing/total_cells*100:.1f}%). "
                    f"Imputation may introduce significant bias."
                ),
                "recommendation": "Verify imputation strategy does not distort distributions",
            })

        # Check for very high cardinality categoricals
        for col, count in profile.unique_counts.items():
            if col in profile.categorical_columns and count > 100:
                findings.append({
                    "severity": Severity.INFO,
                    "category": "data",
                    "message": f"High cardinality categorical: {col} ({count} unique values)",
                    "recommendation": "Consider target encoding or grouping rare categories",
                })

        return findings

    def _review_feature_engineering(self) -> list[dict[str, Any]]:
        """Review feature engineering for potential issues."""
        findings: list[dict[str, Any]] = []

        # Check for potential data leakage — target correlated features
        if self.state.correlations and self.state.problem.target_column:
            target = self.state.problem.target_column
            if target in self.state.correlations:
                target_corrs = self.state.correlations[target]
                for col, corr in target_corrs.items():
                    if col == target:
                        continue
                    if abs(corr) > 0.95:
                        findings.append({
                            "severity": Severity.CRITICAL,
                            "category": "data_leakage",
                            "message": (
                                f"Suspected data leakage: feature '{col}' has "
                                f"correlation {corr:.4f} with target"
                            ),
                            "recommendation": "Remove this feature — it may be a proxy for the target",
                        })

        # Check train/test sizes
        if self.state.X_train is not None and self.state.X_test is not None:
            train_size = len(self.state.X_train)
            test_size = len(self.state.X_test)
            ratio = test_size / (train_size + test_size) if (train_size + test_size) > 0 else 0

            if ratio < 0.1:
                findings.append({
                    "severity": Severity.WARNING,
                    "category": "methodology",
                    "message": f"Test set is very small ({ratio:.1%} of data, {test_size} samples)",
                    "recommendation": "Consider increasing test_size or using cross-validation only",
                })

        # Check feature count vs sample count
        if self.state.X_train is not None:
            n_features = self.state.X_train.shape[1] if hasattr(self.state.X_train, "shape") else 0
            n_samples = len(self.state.X_train)
            if n_features > 0 and n_samples / n_features < 10:
                findings.append({
                    "severity": Severity.WARNING,
                    "category": "methodology",
                    "message": (
                        f"Low sample-to-feature ratio ({n_samples}/{n_features} = "
                        f"{n_samples/n_features:.1f}). Risk of overfitting."
                    ),
                    "recommendation": "Apply dimensionality reduction or feature selection",
                })

        return findings

    def _review_training(self) -> list[dict[str, Any]]:
        """Review training methodology."""
        findings: list[dict[str, Any]] = []

        # Check if cross-validation was used
        cv_scores = self.state.cross_validation_scores
        if not cv_scores:
            findings.append({
                "severity": Severity.WARNING,
                "category": "methodology",
                "message": "No cross-validation scores recorded",
                "recommendation": "Always use cross-validation for reliable performance estimates",
            })

        # Check if only one model was tried
        history = self.state.training_history
        results = history.get("results", [])
        if len(results) < 2:
            findings.append({
                "severity": Severity.INFO,
                "category": "methodology",
                "message": "Only one model was evaluated",
                "recommendation": "Compare at least 2-3 model types for robust selection",
            })

        return findings

    def _review_evaluation(self) -> list[dict[str, Any]]:
        """Review evaluation methodology."""
        findings: list[dict[str, Any]] = []
        metrics = self.state.metrics

        # Check that appropriate metrics were computed
        if not metrics:
            findings.append({
                "severity": Severity.ERROR,
                "category": "evaluation",
                "message": "No evaluation metrics were computed",
                "recommendation": "Compute at least accuracy/F1 or R2/RMSE depending on problem type",
            })

        # Check for suspiciously perfect scores
        for metric_name, value in metrics.items():
            if metric_name in ("accuracy", "f1", "auc_roc", "r2", "f1_weighted") and value >= 0.999:
                findings.append({
                    "severity": Severity.CRITICAL,
                    "category": "data_leakage",
                    "message": (
                        f"Suspiciously perfect {metric_name} = {value:.4f}. "
                        f"This strongly suggests data leakage."
                    ),
                    "recommendation": "Check for target leakage, duplicate rows in train/test, or look-ahead bias",
                })

        return findings

    def _review_methodology(self) -> list[dict[str, Any]]:
        """Review overall methodology and best practices."""
        findings: list[dict[str, Any]] = []

        # Check if baseline was computed
        if not self.state.baseline_metrics:
            findings.append({
                "severity": Severity.INFO,
                "category": "methodology",
                "message": "No baseline metrics computed for comparison",
                "recommendation": "Always compare against a naive baseline",
            })

        # Check reproducibility
        model = self.state.trained_model
        if model and hasattr(model, "random_state") and model.random_state is None:
            findings.append({
                "severity": Severity.WARNING,
                "category": "reproducibility",
                "message": "Model does not have a fixed random_state",
                "recommendation": "Set random_state for reproducible results",
            })

        return findings
