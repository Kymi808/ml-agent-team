"""Training Agent — trains models with cross-validation and hyperparameter tuning."""

from __future__ import annotations

import time
from typing import Any

import numpy as np
from sklearn.model_selection import RandomizedSearchCV, cross_val_score

from ..core.base_agent import BaseAgent
from ..core.messages import AgentMessage
from ..core.types import PipelineStage, ProblemType


def _build_param_distributions(hp_space: dict[str, Any]) -> dict[str, Any]:
    """Convert our hyperparameter space format to sklearn distributions."""
    from scipy import stats

    distributions: dict[str, Any] = {}

    for param, spec in hp_space.items():
        param_type = spec.get("type", "float")

        if param_type == "categorical":
            distributions[param] = spec["choices"]
        elif param_type == "int":
            distributions[param] = stats.randint(spec["low"], spec["high"] + 1)
        elif param_type == "float":
            if spec.get("log", False):
                distributions[param] = stats.loguniform(spec["low"], spec["high"])
            else:
                distributions[param] = stats.uniform(spec["low"], spec["high"] - spec["low"])

    return distributions


class TrainingAgent(BaseAgent):
    """Trains candidate models with cross-validation, hyperparameter tuning, and model comparison."""

    @property
    def stage(self) -> PipelineStage:
        return PipelineStage.TRAINING

    @property
    def description(self) -> str:
        return (
            "Trains all candidate models with cross-validation and optional "
            "hyperparameter tuning, then selects the best performer"
        )

    @property
    def dependencies(self) -> list[PipelineStage]:
        return [PipelineStage.MODEL_BUILDING]

    async def execute(self) -> AgentMessage:
        X_train = self.state.X_train
        y_train = self.state.y_train
        models = self.state.model_pipelines
        hp_spaces = self.state.hyperparameter_spaces
        problem_type = self.state.problem.problem_type

        cv_folds = self.config.params.get("cv_folds", 5)
        do_tuning = self.config.params.get("hyperparameter_tuning", True)
        tuning_iters = self.config.params.get("tuning_iterations", 20)

        scoring = self._get_scoring_metric(problem_type)

        self.logger.info(
            "training_started",
            n_models=len(models),
            scoring=scoring,
            cv_folds=cv_folds,
            tuning=do_tuning,
        )

        results: list[dict[str, Any]] = []

        for model in models:
            model_name = type(model).__name__
            self.logger.info("training_model", model=model_name)
            start_time = time.time()

            try:
                if do_tuning and model_name in hp_spaces and hp_spaces[model_name]:
                    # Hyperparameter tuning with RandomizedSearchCV
                    trained_model, cv_scores, best_params = self._tune_model(
                        model, X_train, y_train, hp_spaces[model_name],
                        scoring, cv_folds, tuning_iters,
                    )
                else:
                    # Train with default params and cross-validate
                    cv_scores = cross_val_score(
                        model, X_train, y_train, cv=cv_folds, scoring=scoring
                    )
                    model.fit(X_train, y_train)
                    trained_model = model
                    best_params = {}

                elapsed = time.time() - start_time

                result = {
                    "name": model_name,
                    "cv_mean": float(np.mean(cv_scores)),
                    "cv_std": float(np.std(cv_scores)),
                    "cv_scores": cv_scores.tolist(),
                    "best_params": best_params,
                    "training_time": round(elapsed, 2),
                    "model": trained_model,
                }
                results.append(result)

                self.logger.info(
                    "model_trained",
                    model=model_name,
                    cv_mean=result["cv_mean"],
                    cv_std=result["cv_std"],
                    time=result["training_time"],
                )

            except Exception as e:
                self.logger.warning("model_training_failed", model=model_name, error=str(e))
                continue

        if not results:
            raise RuntimeError("All models failed to train")

        # Select best model
        results.sort(key=lambda r: r["cv_mean"], reverse=True)
        best = results[0]

        self.state.trained_model = best["model"]
        self.state.best_model_name = best["name"]
        self.state.hyperparameters = best["best_params"]
        self.state.cross_validation_scores = {
            r["name"]: r["cv_scores"] for r in results
        }
        self.state.training_history = {
            "results": [
                {k: v for k, v in r.items() if k != "model"} for r in results
            ],
            "best_model": best["name"],
            "scoring_metric": scoring,
        }

        self.logger.info(
            "training_complete",
            best_model=best["name"],
            best_cv_score=best["cv_mean"],
        )

        return self._result_message({
            "best_model": best["name"],
            "best_cv_mean": round(best["cv_mean"], 4),
            "best_cv_std": round(best["cv_std"], 4),
            "models_trained": len(results),
            "all_results": [
                {"name": r["name"], "cv_mean": round(r["cv_mean"], 4)}
                for r in results
            ],
        })

    def _get_scoring_metric(self, problem_type: ProblemType | None) -> str:
        """Determine the scoring metric based on problem type."""
        configured = self.config.params.get("scoring_metric")
        if configured:
            return configured

        metrics: dict[ProblemType, str] = {
            ProblemType.BINARY_CLASSIFICATION: "f1",
            ProblemType.MULTICLASS_CLASSIFICATION: "f1_macro",
            ProblemType.REGRESSION: "r2",
            ProblemType.CLUSTERING: "adjusted_rand_score",
            ProblemType.ANOMALY_DETECTION: "f1",
        }
        return metrics.get(problem_type, "f1") if problem_type else "f1"

    def _tune_model(
        self,
        model: Any,
        X_train: Any,
        y_train: Any,
        hp_space: dict[str, Any],
        scoring: str,
        cv_folds: int,
        n_iter: int,
    ) -> tuple[Any, Any, dict[str, Any]]:
        """Tune a model using RandomizedSearchCV."""
        param_distributions = _build_param_distributions(hp_space)

        search = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_distributions,
            n_iter=min(n_iter, 50),
            cv=cv_folds,
            scoring=scoring,
            n_jobs=-1,
            random_state=42,
            error_score="raise",
        )

        search.fit(X_train, y_train)

        cv_scores = cross_val_score(
            search.best_estimator_, X_train, y_train, cv=cv_folds, scoring=scoring
        )

        return search.best_estimator_, cv_scores, search.best_params_
