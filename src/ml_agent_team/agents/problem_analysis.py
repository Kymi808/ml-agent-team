"""Problem Analysis Agent — parses the problem, identifies ML task type, defines objectives."""

from __future__ import annotations

import re
from typing import Any

from ..core.base_agent import BaseAgent
from ..core.messages import AgentMessage
from ..core.types import PipelineStage, ProblemType


# Keyword-based heuristics for problem type detection
_PROBLEM_KEYWORDS: dict[ProblemType, list[str]] = {
    ProblemType.BINARY_CLASSIFICATION: [
        "classify", "classification", "binary", "predict whether", "yes or no",
        "true or false", "spam", "fraud", "churn", "default", "positive or negative",
        "detect", "diagnosis",
    ],
    ProblemType.MULTICLASS_CLASSIFICATION: [
        "multiclass", "multi-class", "categorize", "categories", "classify into",
        "multiple classes", "species", "types of",
    ],
    ProblemType.REGRESSION: [
        "predict", "forecast", "regression", "how much", "price", "cost", "value",
        "continuous", "estimate", "amount", "score", "rating", "salary", "revenue",
    ],
    ProblemType.CLUSTERING: [
        "cluster", "segment", "group", "unsupervised", "similarity", "cohort",
    ],
    ProblemType.TIME_SERIES: [
        "time series", "temporal", "forecast", "trend", "seasonal", "stock",
        "demand forecast",
    ],
    ProblemType.ANOMALY_DETECTION: [
        "anomaly", "outlier detection", "novelty", "abnormal", "rare event",
    ],
    ProblemType.RANKING: [
        "rank", "ranking", "order", "relevance", "search",
    ],
    ProblemType.RECOMMENDATION: [
        "recommend", "recommendation", "suggest", "collaborative filtering",
    ],
}


class ProblemAnalysisAgent(BaseAgent):
    """Analyzes the problem description to determine ML task type, objectives, and constraints."""

    @property
    def stage(self) -> PipelineStage:
        return PipelineStage.PROBLEM_ANALYSIS

    @property
    def description(self) -> str:
        return (
            "Parses the problem description, identifies the ML problem type, "
            "defines objectives, success criteria, and constraints"
        )

    @property
    def dependencies(self) -> list[PipelineStage]:
        return []

    async def execute(self) -> AgentMessage:
        problem_desc = self.state.problem.description
        self.logger.info("analyzing_problem", description=problem_desc[:100])

        # Detect problem type from description
        problem_type = self._detect_problem_type(problem_desc)
        self.state.problem.problem_type = problem_type

        # Extract objectives
        self.state.problem.objectives = self._extract_objectives(problem_desc)

        # Set default success criteria based on problem type
        self.state.problem.success_criteria = self._default_success_criteria(problem_type)

        # Detect domain
        self.state.problem.domain = self._detect_domain(problem_desc)

        self.logger.info(
            "problem_analyzed",
            problem_type=problem_type,
            objectives=self.state.problem.objectives,
            domain=self.state.problem.domain,
        )

        return self._result_message({
            "problem_type": problem_type.value,
            "objectives": self.state.problem.objectives,
            "success_criteria": self.state.problem.success_criteria,
            "domain": self.state.problem.domain,
        })

    def _detect_problem_type(self, description: str) -> ProblemType:
        """Detect the ML problem type using keyword matching."""
        desc_lower = description.lower()
        scores: dict[ProblemType, int] = {}

        for ptype, keywords in _PROBLEM_KEYWORDS.items():
            score = sum(1 for kw in keywords if kw in desc_lower)
            if score > 0:
                scores[ptype] = score

        if not scores:
            # Default to binary classification if unclear
            return ProblemType.BINARY_CLASSIFICATION

        return max(scores, key=scores.get)  # type: ignore[arg-type]

    def _extract_objectives(self, description: str) -> list[str]:
        """Extract objectives from the problem description."""
        objectives = []

        # Look for explicit objective statements
        patterns = [
            r"(?:goal|objective|aim|purpose)\s*(?:is|:)\s*(.+?)(?:\.|$)",
            r"(?:we want to|need to|trying to)\s+(.+?)(?:\.|$)",
            r"(?:predict|classify|detect|forecast)\s+(.+?)(?:\.|$)",
        ]

        for pattern in patterns:
            matches = re.findall(pattern, description, re.IGNORECASE)
            objectives.extend(m.strip() for m in matches if m.strip())

        if not objectives:
            objectives = [f"Solve: {description[:200]}"]

        return objectives[:5]  # Cap at 5 objectives

    def _default_success_criteria(self, problem_type: ProblemType) -> dict[str, float]:
        """Set reasonable default success criteria based on problem type."""
        criteria: dict[ProblemType, dict[str, float]] = {
            ProblemType.BINARY_CLASSIFICATION: {
                "accuracy": 0.85,
                "f1_score": 0.80,
                "auc_roc": 0.85,
            },
            ProblemType.MULTICLASS_CLASSIFICATION: {
                "accuracy": 0.80,
                "macro_f1": 0.75,
            },
            ProblemType.REGRESSION: {
                "r2_score": 0.80,
                "mape": 0.15,
            },
            ProblemType.CLUSTERING: {
                "silhouette_score": 0.50,
            },
            ProblemType.TIME_SERIES: {
                "mape": 0.10,
                "rmse_relative": 0.15,
            },
            ProblemType.ANOMALY_DETECTION: {
                "precision": 0.80,
                "recall": 0.75,
            },
            ProblemType.RANKING: {
                "ndcg": 0.80,
            },
            ProblemType.RECOMMENDATION: {
                "precision_at_k": 0.70,
                "recall_at_k": 0.60,
            },
        }
        return criteria.get(problem_type, {"accuracy": 0.80})

    def _detect_domain(self, description: str) -> str:
        """Detect the problem domain from the description."""
        desc_lower = description.lower()
        domains = {
            "healthcare": ["patient", "medical", "health", "disease", "clinical", "diagnosis"],
            "finance": ["financial", "stock", "trading", "credit", "loan", "bank", "fraud"],
            "e-commerce": ["customer", "purchase", "product", "shopping", "cart", "churn"],
            "nlp": ["text", "language", "sentiment", "document", "translation", "nlp"],
            "computer_vision": ["image", "photo", "visual", "object detection", "segmentation"],
            "iot": ["sensor", "iot", "device", "telemetry", "signal"],
        }
        for domain, keywords in domains.items():
            if any(kw in desc_lower for kw in keywords):
                return domain
        return "general"
