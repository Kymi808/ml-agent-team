"""Optimization Agent — applies targeted fixes for diagnosed issues and triggers retraining."""

from __future__ import annotations

from typing import Any

from ..core.base_agent import BaseAgent
from ..core.messages import AgentMessage
from ..core.types import PipelineStage, Severity


class OptimizationAgent(BaseAgent):
    """Applies targeted optimizations based on diagnosis findings and prepares for retraining."""

    @property
    def stage(self) -> PipelineStage:
        return PipelineStage.OPTIMIZATION

    @property
    def description(self) -> str:
        return (
            "Reads diagnosis findings and applies targeted fixes — adjusting regularization, "
            "class weights, feature selection, or model parameters to address identified issues"
        )

    @property
    def dependencies(self) -> list[PipelineStage]:
        return [PipelineStage.DIAGNOSIS]

    async def execute(self) -> AgentMessage:
        issues = self.state.issues
        round_num = self.state.optimization_rounds + 1
        max_rounds = self.state.max_optimization_rounds

        self.logger.info(
            "optimization_started",
            round=round_num,
            max_rounds=max_rounds,
            issues_count=len(issues),
        )

        actions_taken: list[dict[str, str]] = []

        for issue in issues:
            severity = issue.get("severity")
            if severity in (Severity.INFO,):
                continue  # Skip info-level issues

            issue_type = issue.get("type", "")
            action = self._apply_fix(issue_type, issue)
            if action:
                actions_taken.append(action)

        # Increment optimization round
        self.state.optimization_rounds = round_num
        self.state.optimization_history.append({
            "round": round_num,
            "issues_addressed": len(actions_taken),
            "actions": actions_taken,
        })

        # Clear issues for next evaluation round
        self.state.issues = []

        self.logger.info(
            "optimization_complete",
            round=round_num,
            actions_taken=len(actions_taken),
        )

        return self._result_message({
            "round": round_num,
            "actions_taken": len(actions_taken),
            "actions": actions_taken,
            "will_retrain": len(actions_taken) > 0,
        })

    def _apply_fix(self, issue_type: str, issue: dict[str, Any]) -> dict[str, str] | None:
        """Apply a fix for a specific issue type. Returns description of action taken."""
        fix_handlers: dict[str, Any] = {
            "overfitting": self._fix_overfitting,
            "mild_overfitting": self._fix_overfitting,
            "underfitting": self._fix_underfitting,
            "class_imbalance_effect": self._fix_class_imbalance,
            "below_baseline": self._fix_below_baseline,
            "success_criteria_not_met": self._fix_criteria_not_met,
            "high_cv_variance": self._fix_high_variance,
        }

        handler = fix_handlers.get(issue_type)
        if handler:
            return handler(issue)
        return None

    def _fix_overfitting(self, issue: dict[str, Any]) -> dict[str, str]:
        """Address overfitting by increasing regularization or simplifying the model."""
        model = self.state.trained_model

        if hasattr(model, "max_depth") and model.max_depth is not None:
            new_depth = max(2, model.max_depth - 2)
            model.set_params(max_depth=new_depth)
            return {
                "action": f"Reduced max_depth from {model.max_depth} to {new_depth}",
                "target": "overfitting",
            }
        elif hasattr(model, "C"):
            new_C = model.C * 0.1
            model.set_params(C=new_C)
            return {
                "action": f"Reduced regularization parameter C to {new_C}",
                "target": "overfitting",
            }
        elif hasattr(model, "n_estimators"):
            new_n = max(50, model.n_estimators - 50)
            model.set_params(n_estimators=new_n)
            return {
                "action": f"Reduced n_estimators to {new_n}",
                "target": "overfitting",
            }

        return {"action": "No automatic fix applied for overfitting", "target": "overfitting"}

    def _fix_underfitting(self, issue: dict[str, Any]) -> dict[str, str]:
        """Address underfitting by increasing model complexity."""
        model = self.state.trained_model

        if hasattr(model, "max_depth"):
            new_depth = (model.max_depth or 5) + 5
            model.set_params(max_depth=new_depth)
            return {
                "action": f"Increased max_depth to {new_depth}",
                "target": "underfitting",
            }
        elif hasattr(model, "n_estimators"):
            new_n = model.n_estimators + 100
            model.set_params(n_estimators=new_n)
            return {
                "action": f"Increased n_estimators to {new_n}",
                "target": "underfitting",
            }

        return {"action": "No automatic fix applied for underfitting", "target": "underfitting"}

    def _fix_class_imbalance(self, issue: dict[str, Any]) -> dict[str, str]:
        """Address class imbalance by adjusting class weights."""
        model = self.state.trained_model

        if hasattr(model, "class_weight"):
            model.set_params(class_weight="balanced")
            return {
                "action": "Set class_weight='balanced' to handle imbalance",
                "target": "class_imbalance",
            }

        return {
            "action": "Model does not support class_weight parameter",
            "target": "class_imbalance",
        }

    def _fix_below_baseline(self, issue: dict[str, Any]) -> dict[str, str]:
        """Try a fundamentally different approach when below baseline."""
        # Switch to a different candidate model if available
        candidates = self.state.candidate_models
        current_name = self.state.best_model_name

        for candidate in candidates:
            if candidate.get("name") != current_name and candidate.get("model_object"):
                self.state.trained_model = candidate["model_object"]
                self.state.best_model_name = candidate["name"]
                return {
                    "action": f"Switched from {current_name} to {candidate['name']}",
                    "target": "below_baseline",
                }

        return {
            "action": "No alternative model available to switch to",
            "target": "below_baseline",
        }

    def _fix_criteria_not_met(self, issue: dict[str, Any]) -> dict[str, str]:
        """General fix attempt when success criteria are not met."""
        # Increase model capacity
        model = self.state.trained_model

        if hasattr(model, "n_estimators"):
            new_n = model.n_estimators + 100
            model.set_params(n_estimators=new_n)
            return {
                "action": f"Increased n_estimators to {new_n} to improve {issue.get('metric', 'metric')}",
                "target": "criteria_not_met",
            }

        return {
            "action": "Attempted parameter adjustment",
            "target": "criteria_not_met",
        }

    def _fix_high_variance(self, issue: dict[str, Any]) -> dict[str, str]:
        """Address high CV variance."""
        model = self.state.trained_model

        if hasattr(model, "min_samples_leaf"):
            new_val = max(1, getattr(model, "min_samples_leaf", 1)) + 2
            model.set_params(min_samples_leaf=new_val)
            return {
                "action": f"Increased min_samples_leaf to {new_val} for stability",
                "target": "high_variance",
            }

        return {"action": "No automatic fix for high variance", "target": "high_variance"}
