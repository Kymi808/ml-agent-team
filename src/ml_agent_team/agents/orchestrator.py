"""Orchestrator Agent — coordinates the entire pipeline, manages agent lifecycle."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import structlog

from ..core.config import AgentConfig, PipelineConfig
from ..core.message_bus import MessageBus
from ..core.messages import AgentMessage
from ..core.pipeline import Pipeline, PipelineBuilder
from ..core.types import MessageType, PipelineStage
from ..core.workflow_state import WorkflowState
from .data_ingestion import DataIngestionAgent
from .diagnosis import DiagnosisAgent
from .eda import EDAAgent
from .evaluation import EvaluationAgent
from .feature_engineering import FeatureEngineeringAgent
from .literature_review import LiteratureReviewAgent
from .model_building import ModelBuildingAgent
from .model_selection import ModelSelectionAgent
from .optimization import OptimizationAgent
from .peer_review import PeerReviewAgent
from .problem_analysis import ProblemAnalysisAgent
from .reporting import ReportingAgent
from .training import TrainingAgent

logger = structlog.get_logger(__name__)


class OrchestratorAgent:
    """Top-level coordinator that wires together all agents and runs the pipeline.

    The orchestrator:
    - Instantiates all agents based on configuration
    - Builds the pipeline with conditional steps and optimization loops
    - Manages the shared WorkflowState and MessageBus
    - Handles agent failures and decides on retries or halts
    - Provides the main entry point for running the full ML workflow
    """

    def __init__(self, config: PipelineConfig | None = None) -> None:
        self.config = config or PipelineConfig()
        self.message_bus = MessageBus()
        self.state = WorkflowState(max_optimization_rounds=self.config.max_optimization_rounds)
        self.agents: dict[str, Any] = {}
        self._pipeline: Pipeline | None = None

        # Subscribe to errors for monitoring
        self.message_bus.subscribe_topic(MessageType.ERROR, self._handle_error)

    async def run(
        self,
        problem_description: str,
        data_source: str | Path,
    ) -> WorkflowState:
        """Run the full ML agent team pipeline.

        Args:
            problem_description: Natural language description of the ML problem.
            data_source: Path to the data file.

        Returns:
            The final WorkflowState with all artifacts and results.
        """
        logger.info(
            "orchestrator_started",
            problem=problem_description[:100],
            data_source=str(data_source),
        )

        # Initialize state with problem info
        self.state.problem.description = problem_description
        self.state.data_source = str(data_source)

        # Set target column from config if available
        if self.config.data.target_column:
            self.state.problem.target_column = self.config.data.target_column

        # Create agents
        self._create_agents()

        # Build pipeline
        pipeline = self._build_pipeline()

        # Run the pipeline
        try:
            result = await pipeline.run()

            # Run optimization loop if needed
            while result.needs_optimization:
                logger.info(
                    "optimization_loop",
                    round=result.optimization_rounds + 1,
                    max_rounds=result.max_optimization_rounds,
                )
                # Re-run optimization -> training -> evaluation -> diagnosis
                await self.agents["optimization"].run()
                await self.agents["training"].run()
                await self.agents["evaluation"].run()
                await self.agents["diagnosis"].run()

            logger.info(
                "orchestrator_completed",
                is_acceptable=result.is_acceptable,
                stages_completed=len(result.completed_stages),
            )

            return result

        except Exception as e:
            logger.error("orchestrator_failed", error=str(e))
            raise

    def _create_agents(self) -> None:
        """Instantiate all agents based on configuration."""
        output_dir = self.config.output_dir

        agent_defs: list[tuple[str, type, dict[str, Any]]] = [
            ("problem_analysis", ProblemAnalysisAgent, {}),
            ("literature_review", LiteratureReviewAgent, {}),
            ("data_ingestion", DataIngestionAgent, {}),
            ("eda", EDAAgent, {"output_dir": f"{output_dir}/eda"}),
            ("feature_engineering", FeatureEngineeringAgent, {
                "test_size": self.config.data.test_size,
                "random_state": self.config.data.random_state,
            }),
            ("model_selection", ModelSelectionAgent, {}),
            ("model_building", ModelBuildingAgent, {}),
            ("training", TrainingAgent, {
                "cv_folds": self.config.training.cross_validation_folds,
                "hyperparameter_tuning": self.config.training.hyperparameter_tuning,
                "tuning_iterations": self.config.training.tuning_trials,
                "scoring_metric": self.config.training.scoring_metric,
            }),
            ("evaluation", EvaluationAgent, {"output_dir": f"{output_dir}/evaluation"}),
            ("diagnosis", DiagnosisAgent, {}),
            ("optimization", OptimizationAgent, {}),
            ("reporting", ReportingAgent, {"output_dir": output_dir}),
            ("peer_review", PeerReviewAgent, {}),
        ]

        for name, agent_class, extra_params in agent_defs:
            agent_config = self.config.get_agent_config(name)
            # Merge extra params into agent config
            merged_params = {**agent_config.params, **extra_params}
            config = AgentConfig(
                enabled=agent_config.enabled,
                timeout_seconds=agent_config.timeout_seconds,
                retry_on_failure=agent_config.retry_on_failure,
                max_retries=agent_config.max_retries,
                params=merged_params,
            )

            self.agents[name] = agent_class(
                name=name,
                config=config,
                message_bus=self.message_bus,
                workflow_state=self.state,
            )

    def _build_pipeline(self) -> Pipeline:
        """Build the execution pipeline with conditional steps."""
        builder = PipelineBuilder(self.state, self.message_bus)

        # Core pipeline stages
        builder.add_step(self.agents["problem_analysis"])

        # Optional literature review
        if self.config.enable_literature_review:
            builder.add_step(
                self.agents["literature_review"],
                condition=lambda s: self.config.enable_literature_review,
            )

        builder.add_step(self.agents["data_ingestion"])
        builder.add_step(self.agents["eda"])
        builder.add_step(self.agents["feature_engineering"])
        builder.add_step(self.agents["model_selection"])
        builder.add_step(self.agents["model_building"])
        builder.add_step(
            self.agents["training"],
            retry_on_failure=True,
            max_retries=2,
        )
        builder.add_step(self.agents["evaluation"])
        builder.add_step(self.agents["diagnosis"])

        # Reporting always runs (even if not acceptable, to document findings)
        builder.add_step(self.agents["reporting"])

        # Peer review gate
        if self.config.enable_peer_review:
            builder.add_step(self.agents["peer_review"])

        self._pipeline = builder.build()
        return self._pipeline

    async def _handle_error(self, message: AgentMessage) -> None:
        """Handle error messages from agents."""
        logger.error(
            "agent_error_received",
            source=message.source_agent,
            error=message.payload.get("error_message", "Unknown error"),
        )
