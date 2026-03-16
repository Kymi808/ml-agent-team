"""Model Building Agent — constructs model pipelines with hyperparameter search spaces."""

from __future__ import annotations

import importlib
from typing import Any

from ..core.base_agent import BaseAgent
from ..core.messages import AgentMessage
from ..core.types import PipelineStage

# Default hyperparameter search spaces per model class
_HYPERPARAM_SPACES: dict[str, dict[str, Any]] = {
    "LogisticRegression": {
        "C": {"type": "float", "low": 0.001, "high": 100.0, "log": True},
        "penalty": {"type": "categorical", "choices": ["l1", "l2"]},
        "solver": {"type": "categorical", "choices": ["liblinear", "saga"]},
    },
    "RandomForestClassifier": {
        "n_estimators": {"type": "int", "low": 50, "high": 500},
        "max_depth": {"type": "int", "low": 3, "high": 30},
        "min_samples_split": {"type": "int", "low": 2, "high": 20},
        "min_samples_leaf": {"type": "int", "low": 1, "high": 10},
        "max_features": {"type": "categorical", "choices": ["sqrt", "log2", None]},
    },
    "RandomForestRegressor": {
        "n_estimators": {"type": "int", "low": 50, "high": 500},
        "max_depth": {"type": "int", "low": 3, "high": 30},
        "min_samples_split": {"type": "int", "low": 2, "high": 20},
        "min_samples_leaf": {"type": "int", "low": 1, "high": 10},
    },
    "GradientBoostingClassifier": {
        "n_estimators": {"type": "int", "low": 50, "high": 500},
        "max_depth": {"type": "int", "low": 3, "high": 10},
        "learning_rate": {"type": "float", "low": 0.01, "high": 0.3, "log": True},
        "subsample": {"type": "float", "low": 0.6, "high": 1.0},
        "min_samples_split": {"type": "int", "low": 2, "high": 20},
    },
    "GradientBoostingRegressor": {
        "n_estimators": {"type": "int", "low": 50, "high": 500},
        "max_depth": {"type": "int", "low": 3, "high": 10},
        "learning_rate": {"type": "float", "low": 0.01, "high": 0.3, "log": True},
        "subsample": {"type": "float", "low": 0.6, "high": 1.0},
    },
    "SVC": {
        "C": {"type": "float", "low": 0.01, "high": 100.0, "log": True},
        "kernel": {"type": "categorical", "choices": ["rbf", "linear", "poly"]},
        "gamma": {"type": "categorical", "choices": ["scale", "auto"]},
    },
    "Ridge": {
        "alpha": {"type": "float", "low": 0.001, "high": 100.0, "log": True},
    },
    "ElasticNet": {
        "alpha": {"type": "float", "low": 0.001, "high": 10.0, "log": True},
        "l1_ratio": {"type": "float", "low": 0.1, "high": 0.9},
    },
    "KMeans": {
        "n_clusters": {"type": "int", "low": 2, "high": 15},
        "init": {"type": "categorical", "choices": ["k-means++", "random"]},
    },
    "IsolationForest": {
        "n_estimators": {"type": "int", "low": 50, "high": 300},
        "max_features": {"type": "float", "low": 0.5, "high": 1.0},
        "contamination": {"type": "float", "low": 0.01, "high": 0.2},
    },
}


class ModelBuildingAgent(BaseAgent):
    """Constructs model instances and defines hyperparameter search spaces for each candidate."""

    @property
    def stage(self) -> PipelineStage:
        return PipelineStage.MODEL_BUILDING

    @property
    def description(self) -> str:
        return (
            "Instantiates candidate models and configures hyperparameter search spaces "
            "for tuning during the training phase"
        )

    @property
    def dependencies(self) -> list[PipelineStage]:
        return [PipelineStage.MODEL_SELECTION]

    async def execute(self) -> AgentMessage:
        candidates = self.state.candidate_models
        self.logger.info("building_models", n_candidates=len(candidates))

        built_models = []
        hp_spaces: dict[str, dict[str, Any]] = {}

        for candidate in candidates:
            model_name = candidate["name"]
            module_name = candidate["module"]
            class_name = candidate["class"]
            default_params = candidate.get("params", {})

            try:
                # Dynamically import and instantiate the model
                module = importlib.import_module(module_name)
                model_class = getattr(module, class_name)
                model = model_class(**default_params)

                built_models.append(model)

                # Attach hyperparameter space
                space = _HYPERPARAM_SPACES.get(class_name, {})
                hp_spaces[model_name] = space

                self.logger.info(
                    "model_built",
                    model=model_name,
                    hp_space_size=len(space),
                )

            except Exception as e:
                self.logger.warning(
                    "model_build_failed",
                    model=model_name,
                    error=str(e),
                )
                continue

        self.state.model_pipelines = built_models
        self.state.hyperparameter_spaces = hp_spaces

        # Update candidate_models with built status
        for i, model in enumerate(built_models):
            if i < len(candidates):
                candidates[i]["built"] = True
                candidates[i]["model_object"] = model

        self.logger.info("model_building_complete", built=len(built_models))

        return self._result_message(
            {
                "models_built": len(built_models),
                "model_names": [type(m).__name__ for m in built_models],
                "total_hyperparams": sum(len(s) for s in hp_spaces.values()),
            }
        )
