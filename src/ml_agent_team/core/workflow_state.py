"""Shared workflow state (blackboard) that all agents read from and write to."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .types import PipelineStage, ProblemType


@dataclass
class ProblemDefinition:
    """Structured definition of the ML problem."""

    description: str = ""
    problem_type: ProblemType | None = None
    target_column: str | None = None
    objectives: list[str] = field(default_factory=list)
    constraints: list[str] = field(default_factory=list)
    success_criteria: dict[str, float] = field(default_factory=dict)
    domain: str = ""


@dataclass
class DataProfile:
    """Summary statistics and metadata about the dataset."""

    n_rows: int = 0
    n_columns: int = 0
    column_types: dict[str, str] = field(default_factory=dict)
    numeric_columns: list[str] = field(default_factory=list)
    categorical_columns: list[str] = field(default_factory=list)
    datetime_columns: list[str] = field(default_factory=list)
    text_columns: list[str] = field(default_factory=list)
    missing_counts: dict[str, int] = field(default_factory=dict)
    missing_percentages: dict[str, float] = field(default_factory=dict)
    unique_counts: dict[str, int] = field(default_factory=dict)
    summary_stats: dict[str, dict[str, float]] = field(default_factory=dict)


@dataclass
class WorkflowState:
    """The shared blackboard. Every agent reads from and writes to this.

    This is the central data structure that flows through the pipeline.
    Each agent reads what it needs and writes its outputs here for
    downstream agents to consume.
    """

    # Stage tracking
    current_stage: PipelineStage | None = None
    completed_stages: list[PipelineStage] = field(default_factory=list)
    failed_stages: list[PipelineStage] = field(default_factory=list)

    # Problem definition (set by ProblemAnalysisAgent)
    problem: ProblemDefinition = field(default_factory=ProblemDefinition)

    # Literature findings (set by LiteratureReviewAgent)
    literature_findings: list[dict[str, Any]] = field(default_factory=list)
    recommended_approaches: list[str] = field(default_factory=list)

    # Raw data artifacts (set by DataIngestionAgent)
    raw_data: Any = None
    data_source: str = ""
    data_profile: DataProfile = field(default_factory=DataProfile)

    # EDA artifacts (set by EDAAgent)
    eda_report: dict[str, Any] = field(default_factory=dict)
    eda_plots: list[str] = field(default_factory=list)
    correlations: dict[str, dict[str, float]] = field(default_factory=dict)
    outlier_report: dict[str, Any] = field(default_factory=dict)

    # Feature engineering artifacts (set by FeatureEngineeringAgent)
    processed_data: Any = None
    feature_names: list[str] = field(default_factory=list)
    feature_importance: dict[str, float] = field(default_factory=dict)
    encoding_maps: dict[str, dict[str, int]] = field(default_factory=dict)
    X_train: Any = None
    X_test: Any = None
    y_train: Any = None
    y_test: Any = None

    # Model selection artifacts (set by ModelSelectionAgent)
    candidate_models: list[dict[str, Any]] = field(default_factory=list)
    selection_rationale: str = ""

    # Model building artifacts (set by ModelBuildingAgent)
    model_pipelines: list[Any] = field(default_factory=list)
    hyperparameter_spaces: dict[str, dict[str, Any]] = field(default_factory=dict)

    # Training artifacts (set by TrainingAgent)
    trained_model: Any = None
    training_history: dict[str, Any] = field(default_factory=dict)
    hyperparameters: dict[str, Any] = field(default_factory=dict)
    cross_validation_scores: dict[str, list[float]] = field(default_factory=dict)
    best_model_name: str = ""

    # Evaluation artifacts (set by EvaluationAgent)
    metrics: dict[str, float] = field(default_factory=dict)
    baseline_metrics: dict[str, float] = field(default_factory=dict)
    confusion_matrix: Any = None
    evaluation_plots: list[str] = field(default_factory=list)
    classification_report: str = ""

    # Diagnosis artifacts (set by DiagnosisAgent)
    issues: list[dict[str, Any]] = field(default_factory=list)
    bias_report: dict[str, Any] = field(default_factory=dict)
    error_analysis: dict[str, Any] = field(default_factory=dict)
    is_acceptable: bool = False

    # Optimization tracking
    optimization_rounds: int = 0
    max_optimization_rounds: int = 3
    optimization_history: list[dict[str, Any]] = field(default_factory=list)

    # Review artifacts (set by PeerReviewAgent)
    review_findings: list[dict[str, Any]] = field(default_factory=list)
    review_approved: bool = True

    # Final report
    report_path: str | None = None
    report_content: str = ""

    # Generic artifact storage for extensibility
    _artifacts: dict[str, Any] = field(default_factory=dict)

    def get_artifact(self, key: str) -> Any:
        """Get a named artifact from the generic store."""
        return self._artifacts.get(key)

    def set_artifact(self, key: str, value: Any) -> None:
        """Set a named artifact in the generic store."""
        self._artifacts[key] = value

    def mark_stage_completed(self, stage: PipelineStage) -> None:
        """Mark a pipeline stage as completed."""
        if stage not in self.completed_stages:
            self.completed_stages.append(stage)
        if stage in self.failed_stages:
            self.failed_stages.remove(stage)

    def mark_stage_failed(self, stage: PipelineStage) -> None:
        """Mark a pipeline stage as failed."""
        if stage not in self.failed_stages:
            self.failed_stages.append(stage)

    def is_stage_completed(self, stage: PipelineStage) -> bool:
        """Check if a stage has been completed."""
        return stage in self.completed_stages

    @property
    def needs_optimization(self) -> bool:
        """Check if the model needs further optimization."""
        return (
            not self.is_acceptable
            and self.optimization_rounds < self.max_optimization_rounds
            and len(self.issues) > 0
        )
