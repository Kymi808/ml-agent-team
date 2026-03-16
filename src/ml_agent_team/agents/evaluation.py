"""Evaluation Agent — selects metrics, evaluates on test set, generates performance visualizations."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics as sklearn_metrics

from ..core.base_agent import BaseAgent
from ..core.messages import AgentMessage
from ..core.types import PipelineStage, ProblemType


class EvaluationAgent(BaseAgent):
    """Evaluates trained models with appropriate metrics and generates visualizations."""

    @property
    def stage(self) -> PipelineStage:
        return PipelineStage.EVALUATION

    @property
    def description(self) -> str:
        return (
            "Selects appropriate evaluation metrics, evaluates on the test set, "
            "computes baselines, and generates performance visualizations"
        )

    @property
    def dependencies(self) -> list[PipelineStage]:
        return [PipelineStage.TRAINING]

    async def execute(self) -> AgentMessage:
        model = self.state.trained_model
        X_test = self.state.X_test
        y_test = self.state.y_test
        problem_type = self.state.problem.problem_type
        output_dir = Path(self.config.params.get("output_dir", "./output/evaluation"))
        output_dir.mkdir(parents=True, exist_ok=True)

        self.logger.info("evaluating_model", model=self.state.best_model_name)

        # Get predictions
        y_pred = model.predict(X_test)

        # Compute metrics based on problem type
        eval_metrics = self._compute_metrics(y_test, y_pred, model, X_test, problem_type)
        self.state.metrics = eval_metrics

        # Compute baseline metrics
        baseline = self._compute_baseline(y_test, problem_type)
        self.state.baseline_metrics = baseline

        # Generate plots
        plots: list[str] = []

        if problem_type in (
            ProblemType.BINARY_CLASSIFICATION,
            ProblemType.MULTICLASS_CLASSIFICATION,
        ):
            # Confusion matrix
            cm = sklearn_metrics.confusion_matrix(y_test, y_pred)
            self.state.confusion_matrix = cm
            cm_path = self._plot_confusion_matrix(cm, y_test, output_dir)
            if cm_path:
                plots.append(cm_path)

            # Classification report
            self.state.classification_report = sklearn_metrics.classification_report(y_test, y_pred)

            # ROC curve (binary only)
            if problem_type == ProblemType.BINARY_CLASSIFICATION and hasattr(
                model, "predict_proba"
            ):
                roc_path = self._plot_roc_curve(model, X_test, y_test, output_dir)
                if roc_path:
                    plots.append(roc_path)

        elif problem_type == ProblemType.REGRESSION:
            # Residual plot
            res_path = self._plot_residuals(y_test, y_pred, output_dir)
            if res_path:
                plots.append(res_path)

            # Actual vs Predicted
            avp_path = self._plot_actual_vs_predicted(y_test, y_pred, output_dir)
            if avp_path:
                plots.append(avp_path)

        # Model comparison plot
        if self.state.training_history.get("results"):
            comp_path = self._plot_model_comparison(
                self.state.training_history["results"], output_dir
            )
            if comp_path:
                plots.append(comp_path)

        self.state.evaluation_plots = plots

        # Determine if results meet success criteria
        meets_criteria = self._check_success_criteria(eval_metrics)

        self.logger.info(
            "evaluation_complete",
            metrics=eval_metrics,
            baseline=baseline,
            meets_criteria=meets_criteria,
        )

        return self._result_message(
            {
                "metrics": eval_metrics,
                "baseline_metrics": baseline,
                "meets_success_criteria": meets_criteria,
                "plots_generated": len(plots),
            }
        )

    def _compute_metrics(
        self,
        y_test: Any,
        y_pred: Any,
        model: Any,
        X_test: Any,
        problem_type: ProblemType | None,
    ) -> dict[str, float]:
        """Compute evaluation metrics based on problem type."""
        result: dict[str, float] = {}

        if problem_type in (
            ProblemType.BINARY_CLASSIFICATION,
            ProblemType.MULTICLASS_CLASSIFICATION,
        ):
            result["accuracy"] = float(sklearn_metrics.accuracy_score(y_test, y_pred))
            result["f1_weighted"] = float(
                sklearn_metrics.f1_score(y_test, y_pred, average="weighted", zero_division=0)
            )
            result["precision_weighted"] = float(
                sklearn_metrics.precision_score(y_test, y_pred, average="weighted", zero_division=0)
            )
            result["recall_weighted"] = float(
                sklearn_metrics.recall_score(y_test, y_pred, average="weighted", zero_division=0)
            )

            if problem_type == ProblemType.BINARY_CLASSIFICATION:
                result["f1"] = float(sklearn_metrics.f1_score(y_test, y_pred, zero_division=0))
                if hasattr(model, "predict_proba"):
                    y_proba = model.predict_proba(X_test)[:, 1]
                    result["auc_roc"] = float(sklearn_metrics.roc_auc_score(y_test, y_proba))
            else:
                result["f1_macro"] = float(
                    sklearn_metrics.f1_score(y_test, y_pred, average="macro", zero_division=0)
                )

        elif problem_type == ProblemType.REGRESSION:
            result["r2"] = float(sklearn_metrics.r2_score(y_test, y_pred))
            result["rmse"] = float(np.sqrt(sklearn_metrics.mean_squared_error(y_test, y_pred)))
            result["mae"] = float(sklearn_metrics.mean_absolute_error(y_test, y_pred))
            y_test_arr = np.array(y_test)
            nonzero = y_test_arr != 0
            if nonzero.any():
                result["mape"] = float(
                    np.mean(
                        np.abs(
                            (y_test_arr[nonzero] - np.array(y_pred)[nonzero]) / y_test_arr[nonzero]
                        )
                    )
                )

        return {k: round(v, 4) for k, v in result.items()}

    def _compute_baseline(self, y_test: Any, problem_type: ProblemType | None) -> dict[str, float]:
        """Compute baseline metrics (majority class / mean predictor)."""
        result: dict[str, float] = {}
        y_arr = np.array(y_test)

        if problem_type in (
            ProblemType.BINARY_CLASSIFICATION,
            ProblemType.MULTICLASS_CLASSIFICATION,
        ):
            # Majority class baseline
            from collections import Counter

            most_common = Counter(y_arr).most_common(1)[0][0]
            baseline_pred = np.full_like(y_arr, most_common)
            result["baseline_accuracy"] = float(
                sklearn_metrics.accuracy_score(y_arr, baseline_pred)
            )

        elif problem_type == ProblemType.REGRESSION:
            # Mean predictor baseline
            mean_pred = np.full_like(y_arr, y_arr.mean(), dtype=float)
            result["baseline_rmse"] = float(
                np.sqrt(sklearn_metrics.mean_squared_error(y_arr, mean_pred))
            )
            result["baseline_r2"] = 0.0  # By definition

        return {k: round(v, 4) for k, v in result.items()}

    def _check_success_criteria(self, metrics: dict[str, float]) -> bool:
        """Check if metrics meet the success criteria defined in problem analysis."""
        criteria = self.state.problem.success_criteria
        if not criteria:
            return True

        for metric_name, threshold in criteria.items():
            if metric_name in metrics:
                if metric_name in ("mape", "rmse", "mae"):
                    # Lower is better
                    if metrics[metric_name] > threshold:
                        return False
                else:
                    # Higher is better
                    if metrics[metric_name] < threshold:
                        return False
        return True

    # ── Plotting methods ──

    def _plot_confusion_matrix(self, cm: Any, y_test: Any, output_dir: Path) -> str | None:
        """Plot confusion matrix heatmap."""
        fig, ax = plt.subplots(figsize=(8, 6))
        disp = sklearn_metrics.ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=np.unique(y_test),
        )
        disp.plot(cmap="Blues", ax=ax)
        ax.set_title("Confusion Matrix")
        plt.tight_layout()

        path = str(output_dir / "confusion_matrix.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        return path

    def _plot_roc_curve(self, model: Any, X_test: Any, y_test: Any, output_dir: Path) -> str | None:
        """Plot ROC curve."""
        y_proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = sklearn_metrics.roc_curve(y_test, y_proba)
        auc = sklearn_metrics.roc_auc_score(y_test, y_proba)

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(fpr, tpr, label=f"AUC = {auc:.4f}", linewidth=2)
        ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curve")
        ax.legend()
        plt.tight_layout()

        path = str(output_dir / "roc_curve.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        return path

    def _plot_residuals(self, y_test: Any, y_pred: Any, output_dir: Path) -> str | None:
        """Plot residuals for regression."""
        residuals = np.array(y_test) - np.array(y_pred)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Residual scatter
        axes[0].scatter(y_pred, residuals, alpha=0.5, s=10)
        axes[0].axhline(y=0, color="r", linestyle="--")
        axes[0].set_xlabel("Predicted")
        axes[0].set_ylabel("Residual")
        axes[0].set_title("Residual Plot")

        # Residual histogram
        axes[1].hist(residuals, bins=30, edgecolor="white")
        axes[1].set_xlabel("Residual")
        axes[1].set_ylabel("Count")
        axes[1].set_title("Residual Distribution")

        plt.tight_layout()
        path = str(output_dir / "residuals.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        return path

    def _plot_actual_vs_predicted(self, y_test: Any, y_pred: Any, output_dir: Path) -> str | None:
        """Plot actual vs predicted scatter."""
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(y_test, y_pred, alpha=0.5, s=10)
        min_val = min(np.min(y_test), np.min(y_pred))
        max_val = max(np.max(y_test), np.max(y_pred))
        ax.plot([min_val, max_val], [min_val, max_val], "r--", label="Perfect")
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
        ax.set_title("Actual vs Predicted")
        ax.legend()
        plt.tight_layout()

        path = str(output_dir / "actual_vs_predicted.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        return path

    def _plot_model_comparison(self, results: list[dict[str, Any]], output_dir: Path) -> str | None:
        """Plot bar chart comparing model cross-validation scores."""
        names = [r["name"] for r in results]
        means = [r["cv_mean"] for r in results]
        stds = [r["cv_std"] for r in results]

        fig, ax = plt.subplots(figsize=(10, 5))
        bars = ax.bar(names, means, yerr=stds, capsize=5, color="steelblue", edgecolor="white")
        ax.set_ylabel("CV Score")
        ax.set_title("Model Comparison (Cross-Validation)")
        ax.tick_params(axis="x", rotation=45)

        # Highlight best
        best_idx = np.argmax(means)
        bars[best_idx].set_color("forestgreen")

        plt.tight_layout()
        path = str(output_dir / "model_comparison.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        return path
