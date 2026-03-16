"""Model Selection Agent — recommends candidate models based on problem type and data characteristics."""

from __future__ import annotations

from typing import Any

from ..core.base_agent import BaseAgent
from ..core.messages import AgentMessage
from ..core.types import PipelineStage, ProblemType

# Model candidates organized by problem type
_MODEL_CATALOG: dict[ProblemType, list[dict[str, Any]]] = {
    ProblemType.BINARY_CLASSIFICATION: [
        {
            "name": "LogisticRegression",
            "module": "sklearn.linear_model",
            "class": "LogisticRegression",
            "rationale": "Strong interpretable baseline with regularization",
            "complexity": "low",
            "params": {"max_iter": 1000, "random_state": 42},
        },
        {
            "name": "RandomForest",
            "module": "sklearn.ensemble",
            "class": "RandomForestClassifier",
            "rationale": "Robust ensemble, good feature importance, handles non-linearity",
            "complexity": "medium",
            "params": {"n_estimators": 100, "random_state": 42},
        },
        {
            "name": "GradientBoosting",
            "module": "sklearn.ensemble",
            "class": "GradientBoostingClassifier",
            "rationale": "High performance on tabular data, sequential error correction",
            "complexity": "medium",
            "params": {"n_estimators": 100, "random_state": 42},
        },
        {
            "name": "SVM",
            "module": "sklearn.svm",
            "class": "SVC",
            "rationale": "Effective in high-dimensional spaces, strong margins",
            "complexity": "medium",
            "params": {"probability": True, "random_state": 42},
        },
    ],
    ProblemType.MULTICLASS_CLASSIFICATION: [
        {
            "name": "LogisticRegression",
            "module": "sklearn.linear_model",
            "class": "LogisticRegression",
            "rationale": "Baseline with multi-class support (OvR or multinomial)",
            "complexity": "low",
            "params": {"max_iter": 1000, "multi_class": "multinomial", "random_state": 42},
        },
        {
            "name": "RandomForest",
            "module": "sklearn.ensemble",
            "class": "RandomForestClassifier",
            "rationale": "Naturally handles multi-class, robust performance",
            "complexity": "medium",
            "params": {"n_estimators": 100, "random_state": 42},
        },
        {
            "name": "GradientBoosting",
            "module": "sklearn.ensemble",
            "class": "GradientBoostingClassifier",
            "rationale": "Strong multi-class performance on tabular data",
            "complexity": "medium",
            "params": {"n_estimators": 100, "random_state": 42},
        },
    ],
    ProblemType.REGRESSION: [
        {
            "name": "LinearRegression",
            "module": "sklearn.linear_model",
            "class": "Ridge",
            "rationale": "Regularized linear baseline, good when features are correlated",
            "complexity": "low",
            "params": {"alpha": 1.0},
        },
        {
            "name": "ElasticNet",
            "module": "sklearn.linear_model",
            "class": "ElasticNet",
            "rationale": "Combines L1/L2 regularization for feature selection and stability",
            "complexity": "low",
            "params": {"random_state": 42},
        },
        {
            "name": "RandomForest",
            "module": "sklearn.ensemble",
            "class": "RandomForestRegressor",
            "rationale": "Non-linear regression with built-in feature importance",
            "complexity": "medium",
            "params": {"n_estimators": 100, "random_state": 42},
        },
        {
            "name": "GradientBoosting",
            "module": "sklearn.ensemble",
            "class": "GradientBoostingRegressor",
            "rationale": "Top performer for tabular regression tasks",
            "complexity": "medium",
            "params": {"n_estimators": 100, "random_state": 42},
        },
    ],
    ProblemType.CLUSTERING: [
        {
            "name": "KMeans",
            "module": "sklearn.cluster",
            "class": "KMeans",
            "rationale": "Fast, scalable, good for spherical clusters",
            "complexity": "low",
            "params": {"n_init": 10, "random_state": 42},
        },
        {
            "name": "DBSCAN",
            "module": "sklearn.cluster",
            "class": "DBSCAN",
            "rationale": "Density-based, discovers arbitrary-shaped clusters",
            "complexity": "medium",
            "params": {},
        },
    ],
    ProblemType.ANOMALY_DETECTION: [
        {
            "name": "IsolationForest",
            "module": "sklearn.ensemble",
            "class": "IsolationForest",
            "rationale": "Efficient tree-based anomaly detection",
            "complexity": "medium",
            "params": {"random_state": 42, "contamination": "auto"},
        },
        {
            "name": "LocalOutlierFactor",
            "module": "sklearn.neighbors",
            "class": "LocalOutlierFactor",
            "rationale": "Density-based local anomaly detection",
            "complexity": "medium",
            "params": {"novelty": True},
        },
    ],
}


class ModelSelectionAgent(BaseAgent):
    """Recommends candidate models based on problem type, data size, and literature findings."""

    @property
    def stage(self) -> PipelineStage:
        return PipelineStage.MODEL_SELECTION

    @property
    def description(self) -> str:
        return (
            "Selects candidate models based on problem type, dataset characteristics, "
            "and recommendations from literature review"
        )

    @property
    def dependencies(self) -> list[PipelineStage]:
        return [PipelineStage.FEATURE_ENGINEERING]

    async def execute(self) -> AgentMessage:
        problem_type = self.state.problem.problem_type
        n_samples = len(self.state.X_train) if self.state.X_train is not None else 0
        n_features = len(self.state.feature_names)

        self.logger.info(
            "selecting_models",
            problem_type=problem_type,
            n_samples=n_samples,
            n_features=n_features,
        )

        # Get base candidates for this problem type
        candidates = _MODEL_CATALOG.get(
            problem_type, _MODEL_CATALOG[ProblemType.BINARY_CLASSIFICATION]
        )
        candidates = [dict(c) for c in candidates]  # Copy to avoid mutating catalog

        # Filter based on data size
        candidates = self._filter_by_data_size(candidates, n_samples, n_features)

        # Rank candidates
        candidates = self._rank_candidates(candidates, n_samples, n_features)

        # Build rationale
        rationale = self._build_rationale(candidates, problem_type, n_samples, n_features)

        self.state.candidate_models = candidates
        self.state.selection_rationale = rationale

        self.logger.info(
            "models_selected",
            candidates=[c["name"] for c in candidates],
        )

        return self._result_message(
            {
                "candidates": [c["name"] for c in candidates],
                "rationale": rationale,
            }
        )

    def _filter_by_data_size(
        self, candidates: list[dict[str, Any]], n_samples: int, n_features: int
    ) -> list[dict[str, Any]]:
        """Remove models that don't suit the dataset size."""
        filtered = []
        for c in candidates:
            # SVM is slow on large datasets
            if c["name"] == "SVM" and n_samples > 50000:
                continue
            filtered.append(c)
        return filtered

    def _rank_candidates(
        self, candidates: list[dict[str, Any]], n_samples: int, n_features: int
    ) -> list[dict[str, Any]]:
        """Rank candidates by expected suitability."""
        complexity_order = {"low": 1, "medium": 2, "high": 3}

        def score(c: dict[str, Any]) -> float:
            # Prefer medium complexity for most cases
            complexity = complexity_order.get(c.get("complexity", "medium"), 2)
            if n_samples < 1000:
                # Prefer simpler models for small datasets
                return -complexity
            else:
                # Prefer more complex models for larger datasets
                return complexity

        candidates.sort(key=score, reverse=True)
        return candidates

    def _build_rationale(
        self,
        candidates: list[dict[str, Any]],
        problem_type: ProblemType | None,
        n_samples: int,
        n_features: int,
    ) -> str:
        """Build a human-readable rationale for the selection."""
        parts = [
            f"Problem type: {problem_type}",
            f"Dataset: {n_samples} samples, {n_features} features",
            f"Selected {len(candidates)} candidate(s):",
        ]
        for i, c in enumerate(candidates, 1):
            parts.append(f"  {i}. {c['name']} — {c['rationale']}")
        return "\n".join(parts)
