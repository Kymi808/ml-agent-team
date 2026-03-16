"""Diagnosis Agent — identifies model weaknesses, error patterns, and bias."""

from __future__ import annotations

from typing import Any

import numpy as np

from ..core.base_agent import BaseAgent
from ..core.messages import AgentMessage
from ..core.types import PipelineStage, ProblemType, Severity


class DiagnosisAgent(BaseAgent):
    """Diagnoses model issues: overfitting, underfitting, bias, calibration, error patterns."""

    @property
    def stage(self) -> PipelineStage:
        return PipelineStage.DIAGNOSIS

    @property
    def description(self) -> str:
        return (
            "Analyzes model performance to identify weaknesses — overfitting, underfitting, "
            "class imbalance effects, error patterns, and bias"
        )

    @property
    def dependencies(self) -> list[PipelineStage]:
        return [PipelineStage.EVALUATION]

    async def execute(self) -> AgentMessage:
        model = self.state.trained_model
        metrics = self.state.metrics
        baseline = self.state.baseline_metrics
        problem_type = self.state.problem.problem_type
        cv_scores = self.state.cross_validation_scores.get(self.state.best_model_name, [])

        self.logger.info("diagnosing_model", model=self.state.best_model_name)

        issues: list[dict[str, Any]] = []

        # 1. Check for overfitting / underfitting
        issues.extend(self._check_fitting(model, metrics, cv_scores, problem_type))

        # 2. Check if model beats baseline
        issues.extend(self._check_vs_baseline(metrics, baseline, problem_type))

        # 3. Check success criteria
        issues.extend(self._check_success_criteria(metrics))

        # 4. Check class imbalance effects (classification)
        if problem_type in (ProblemType.BINARY_CLASSIFICATION, ProblemType.MULTICLASS_CLASSIFICATION):
            issues.extend(self._check_class_imbalance(metrics))

        # 5. Check CV variance
        if cv_scores:
            issues.extend(self._check_cv_stability(cv_scores))

        # 6. Error analysis
        error_analysis = self._analyze_errors(problem_type)
        self.state.error_analysis = error_analysis

        # Determine if acceptable
        critical_issues = [i for i in issues if i["severity"] == Severity.CRITICAL]
        error_issues = [i for i in issues if i["severity"] == Severity.ERROR]
        self.state.is_acceptable = len(critical_issues) == 0 and len(error_issues) == 0
        self.state.issues = issues

        self.logger.info(
            "diagnosis_complete",
            total_issues=len(issues),
            critical=len(critical_issues),
            errors=len(error_issues),
            is_acceptable=self.state.is_acceptable,
        )

        return self._result_message({
            "total_issues": len(issues),
            "critical_issues": len(critical_issues),
            "error_issues": len(error_issues),
            "is_acceptable": self.state.is_acceptable,
            "issues_summary": [
                {"severity": i["severity"], "description": i["description"]}
                for i in issues
            ],
        })

    def _check_fitting(
        self, model: Any, metrics: dict[str, float],
        cv_scores: list[float], problem_type: ProblemType | None,
    ) -> list[dict[str, Any]]:
        """Check for overfitting or underfitting."""
        issues: list[dict[str, Any]] = []

        if not cv_scores:
            return issues

        cv_mean = float(np.mean(cv_scores))

        # Get test score for comparison
        if problem_type in (ProblemType.BINARY_CLASSIFICATION, ProblemType.MULTICLASS_CLASSIFICATION):
            test_score = metrics.get("f1_weighted", metrics.get("accuracy", 0))
        else:
            test_score = metrics.get("r2", 0)

        # Overfitting: train/CV score much higher than test score
        gap = cv_mean - test_score
        if gap > 0.1:
            issues.append({
                "type": "overfitting",
                "severity": Severity.ERROR,
                "description": (
                    f"Possible overfitting detected: CV score ({cv_mean:.4f}) is significantly "
                    f"higher than test score ({test_score:.4f}), gap = {gap:.4f}"
                ),
                "recommendation": "Increase regularization, reduce model complexity, or add more training data",
                "cv_score": cv_mean,
                "test_score": test_score,
            })
        elif gap > 0.05:
            issues.append({
                "type": "mild_overfitting",
                "severity": Severity.WARNING,
                "description": (
                    f"Mild overfitting: CV score ({cv_mean:.4f}) slightly higher than "
                    f"test score ({test_score:.4f}), gap = {gap:.4f}"
                ),
                "recommendation": "Monitor but may be acceptable",
            })

        # Underfitting: both scores are low
        threshold = 0.5
        if cv_mean < threshold and test_score < threshold:
            issues.append({
                "type": "underfitting",
                "severity": Severity.ERROR,
                "description": (
                    f"Possible underfitting: both CV ({cv_mean:.4f}) and test ({test_score:.4f}) "
                    f"scores are below {threshold}"
                ),
                "recommendation": "Increase model complexity, add more features, or reduce regularization",
            })

        return issues

    def _check_vs_baseline(
        self, metrics: dict[str, float], baseline: dict[str, float],
        problem_type: ProblemType | None,
    ) -> list[dict[str, Any]]:
        """Check if the model meaningfully beats the baseline."""
        issues: list[dict[str, Any]] = []

        if problem_type in (ProblemType.BINARY_CLASSIFICATION, ProblemType.MULTICLASS_CLASSIFICATION):
            model_acc = metrics.get("accuracy", 0)
            baseline_acc = baseline.get("baseline_accuracy", 0)

            if model_acc <= baseline_acc:
                issues.append({
                    "type": "below_baseline",
                    "severity": Severity.CRITICAL,
                    "description": (
                        f"Model accuracy ({model_acc:.4f}) does not beat majority class "
                        f"baseline ({baseline_acc:.4f})"
                    ),
                    "recommendation": "Model is not learning meaningful patterns — review features and data quality",
                })
            elif model_acc - baseline_acc < 0.02:
                issues.append({
                    "type": "marginal_improvement",
                    "severity": Severity.WARNING,
                    "description": (
                        f"Model barely beats baseline: {model_acc:.4f} vs {baseline_acc:.4f}"
                    ),
                    "recommendation": "Consider whether the model adds sufficient value over a simple rule",
                })

        elif problem_type == ProblemType.REGRESSION:
            r2 = metrics.get("r2", 0)
            if r2 < 0:
                issues.append({
                    "type": "below_baseline",
                    "severity": Severity.CRITICAL,
                    "description": f"Negative R2 ({r2:.4f}) — model is worse than predicting the mean",
                    "recommendation": "Fundamental modeling issue — review features and approach",
                })

        return issues

    def _check_success_criteria(self, metrics: dict[str, float]) -> list[dict[str, Any]]:
        """Check if success criteria from problem analysis are met."""
        issues: list[dict[str, Any]] = []
        criteria = self.state.problem.success_criteria

        for metric_name, threshold in criteria.items():
            if metric_name not in metrics:
                continue

            value = metrics[metric_name]
            lower_better = metric_name in ("mape", "rmse", "mae")

            if lower_better:
                met = value <= threshold
            else:
                met = value >= threshold

            if not met:
                direction = "below" if not lower_better else "above"
                issues.append({
                    "type": "success_criteria_not_met",
                    "severity": Severity.ERROR,
                    "description": (
                        f"Success criterion not met: {metric_name} = {value:.4f}, "
                        f"required {'<=' if lower_better else '>='} {threshold}"
                    ),
                    "recommendation": f"Optimize model to bring {metric_name} {direction} {threshold}",
                    "metric": metric_name,
                    "value": value,
                    "threshold": threshold,
                })

        return issues

    def _check_class_imbalance(self, metrics: dict[str, float]) -> list[dict[str, Any]]:
        """Check for class imbalance effects in classification."""
        issues: list[dict[str, Any]] = []

        accuracy = metrics.get("accuracy", 0)
        f1 = metrics.get("f1_weighted", metrics.get("f1", 0))

        # Large gap between accuracy and F1 suggests imbalance issues
        if accuracy > 0 and f1 > 0 and abs(accuracy - f1) > 0.1:
            issues.append({
                "type": "class_imbalance_effect",
                "severity": Severity.WARNING,
                "description": (
                    f"Gap between accuracy ({accuracy:.4f}) and F1 ({f1:.4f}) suggests "
                    f"class imbalance may be affecting results"
                ),
                "recommendation": "Consider class weights, SMOTE, or threshold tuning",
            })

        return issues

    def _check_cv_stability(self, cv_scores: list[float]) -> list[dict[str, Any]]:
        """Check if cross-validation scores are stable."""
        issues: list[dict[str, Any]] = []

        cv_std = float(np.std(cv_scores))
        cv_mean = float(np.mean(cv_scores))

        if cv_mean > 0 and cv_std / cv_mean > 0.15:
            issues.append({
                "type": "high_cv_variance",
                "severity": Severity.WARNING,
                "description": (
                    f"High CV variance: std/mean = {cv_std/cv_mean:.2f} "
                    f"(mean={cv_mean:.4f}, std={cv_std:.4f})"
                ),
                "recommendation": "Model may be unstable — consider more robust models or more data",
            })

        return issues

    def _analyze_errors(self, problem_type: ProblemType | None) -> dict[str, Any]:
        """Perform error analysis on predictions."""
        analysis: dict[str, Any] = {}

        if self.state.confusion_matrix is not None:
            cm = np.array(self.state.confusion_matrix)
            # Per-class error rates
            per_class_accuracy = cm.diagonal() / cm.sum(axis=1)
            analysis["per_class_accuracy"] = per_class_accuracy.tolist()
            analysis["worst_class_idx"] = int(np.argmin(per_class_accuracy))
            analysis["worst_class_accuracy"] = float(np.min(per_class_accuracy))

        return analysis
