"""Literature Review Agent — researches relevant approaches and state-of-the-art methods."""

from __future__ import annotations

from ..core.base_agent import BaseAgent
from ..core.messages import AgentMessage
from ..core.types import PipelineStage, ProblemType

# Knowledge base of well-known approaches per problem type
_APPROACH_KNOWLEDGE: dict[ProblemType, list[dict[str, str]]] = {
    ProblemType.BINARY_CLASSIFICATION: [
        {
            "name": "Gradient Boosted Trees (XGBoost/LightGBM)",
            "description": "State-of-the-art for tabular data. Handles mixed feature types, missing values, and feature interactions well.",
            "when_to_use": "Default choice for structured/tabular data with moderate to large datasets.",
            "reference": "Chen & Guestrin, 2016 - XGBoost: A Scalable Tree Boosting System",
        },
        {
            "name": "Logistic Regression with regularization",
            "description": "Strong baseline with interpretability. L1 for feature selection, L2 for correlated features.",
            "when_to_use": "When interpretability is important or as a baseline.",
            "reference": "Standard statistical method",
        },
        {
            "name": "Random Forest",
            "description": "Robust ensemble method. Less prone to overfitting than single trees. Good feature importance.",
            "when_to_use": "Medium-sized datasets where robustness matters more than peak performance.",
            "reference": "Breiman, 2001 - Random Forests",
        },
        {
            "name": "Neural Networks (MLP/TabNet)",
            "description": "Can capture complex non-linear relationships. TabNet offers attention-based feature selection.",
            "when_to_use": "Large datasets with complex patterns. TabNet when interpretability of deep models is needed.",
            "reference": "Arik & Pfister, 2021 - TabNet",
        },
    ],
    ProblemType.MULTICLASS_CLASSIFICATION: [
        {
            "name": "Gradient Boosted Trees with multi-class objective",
            "description": "Extend binary boosting to multi-class via softmax or one-vs-rest.",
            "when_to_use": "Default for tabular multi-class problems.",
            "reference": "Same as binary, with multi:softmax objective",
        },
        {
            "name": "Random Forest",
            "description": "Naturally handles multi-class. Good baseline with feature importance.",
            "when_to_use": "When model stability is prioritized.",
            "reference": "Breiman, 2001",
        },
    ],
    ProblemType.REGRESSION: [
        {
            "name": "Gradient Boosted Regression Trees",
            "description": "Top performer for tabular regression. Handles non-linearities and interactions.",
            "when_to_use": "Default for structured regression problems.",
            "reference": "Friedman, 2001 - Greedy Function Approximation",
        },
        {
            "name": "Regularized Linear Models (Ridge/Lasso/ElasticNet)",
            "description": "Interpretable baseline. ElasticNet combines L1/L2 for feature selection + stability.",
            "when_to_use": "When linear relationships dominate or interpretability is required.",
            "reference": "Zou & Hastie, 2005 - Regularization and Variable Selection via the Elastic Net",
        },
        {
            "name": "Random Forest Regressor",
            "description": "Robust to outliers and non-linear relationships. Natural uncertainty estimates via tree variance.",
            "when_to_use": "Medium-sized datasets, especially with outliers.",
            "reference": "Breiman, 2001",
        },
    ],
    ProblemType.CLUSTERING: [
        {
            "name": "K-Means / K-Means++",
            "description": "Fast, scalable clustering. K-Means++ improves initialization.",
            "when_to_use": "Spherical clusters, known k, large datasets.",
            "reference": "Arthur & Vassilvitskii, 2007 - K-Means++",
        },
        {
            "name": "DBSCAN",
            "description": "Density-based clustering. Discovers arbitrary-shaped clusters and identifies noise.",
            "when_to_use": "Unknown number of clusters, non-spherical shapes, noise present.",
            "reference": "Ester et al., 1996",
        },
    ],
    ProblemType.TIME_SERIES: [
        {
            "name": "Prophet / ARIMA",
            "description": "Statistical time series models. Prophet handles seasonality and holidays automatically.",
            "when_to_use": "Single time series forecasting with clear seasonal patterns.",
            "reference": "Taylor & Letham, 2018 - Forecasting at Scale",
        },
        {
            "name": "LightGBM with lag features",
            "description": "Gradient boosting on engineered time features (lags, rolling stats, date features).",
            "when_to_use": "Multiple time series or when external features are available.",
            "reference": "Standard ML approach to time series",
        },
    ],
    ProblemType.ANOMALY_DETECTION: [
        {
            "name": "Isolation Forest",
            "description": "Tree-based anomaly detection. Efficient for high-dimensional data.",
            "when_to_use": "General-purpose anomaly detection on tabular data.",
            "reference": "Liu et al., 2008 - Isolation Forest",
        },
        {
            "name": "Autoencoder-based detection",
            "description": "Learn normal pattern representation; anomalies have high reconstruction error.",
            "when_to_use": "Complex data where learned representations are needed.",
            "reference": "Standard deep learning approach",
        },
    ],
}


class LiteratureReviewAgent(BaseAgent):
    """Researches relevant approaches and state-of-the-art methods for the problem."""

    @property
    def stage(self) -> PipelineStage:
        return PipelineStage.LITERATURE_REVIEW

    @property
    def description(self) -> str:
        return (
            "Reviews relevant ML literature and recommends state-of-the-art "
            "approaches based on the identified problem type"
        )

    @property
    def dependencies(self) -> list[PipelineStage]:
        return [PipelineStage.PROBLEM_ANALYSIS]

    async def execute(self) -> AgentMessage:
        problem_type = self.state.problem.problem_type
        self.logger.info("reviewing_literature", problem_type=problem_type)

        # Look up relevant approaches
        approaches = _APPROACH_KNOWLEDGE.get(problem_type, [])

        # Also include general best practices
        general_practices = [
            {
                "name": "Cross-validation strategy",
                "description": "Use stratified k-fold for classification, standard k-fold for regression. "
                "Time-based splits for temporal data.",
                "when_to_use": "Always",
                "reference": "Standard ML practice",
            },
            {
                "name": "Baseline model",
                "description": "Always establish a simple baseline (majority class, mean predictor) "
                "to contextualize results.",
                "when_to_use": "Always as first step in evaluation",
                "reference": "Standard ML practice",
            },
            {
                "name": "Feature importance analysis",
                "description": "Use SHAP values or permutation importance for model-agnostic feature understanding.",
                "when_to_use": "After model training for interpretability",
                "reference": "Lundberg & Lee, 2017 - SHAP",
            },
        ]

        all_findings = approaches + general_practices
        self.state.literature_findings = all_findings
        self.state.recommended_approaches = [a["name"] for a in approaches[:3]]

        self.logger.info(
            "literature_review_complete",
            findings_count=len(all_findings),
            top_approaches=self.state.recommended_approaches,
        )

        return self._result_message(
            {
                "findings_count": len(all_findings),
                "recommended_approaches": self.state.recommended_approaches,
            }
        )
