"""Pipeline engine for orchestrating sequential agent execution."""

from __future__ import annotations

from collections.abc import Callable

import structlog

from .base_agent import BaseAgent
from .message_bus import MessageBus
from .types import AgentStatus, PipelineStage
from .workflow_state import WorkflowState

logger = structlog.get_logger(__name__)


class PipelineStep:
    """A single step in the pipeline, wrapping an agent with conditions."""

    def __init__(
        self,
        agent: BaseAgent,
        condition: Callable[[WorkflowState], bool] | None = None,
        retry_on_failure: bool = False,
        max_retries: int = 1,
    ) -> None:
        self.agent = agent
        self.condition = condition
        self.retry_on_failure = retry_on_failure
        self.max_retries = max_retries

    def should_run(self, state: WorkflowState) -> bool:
        """Check if this step should execute given the current state."""
        if self.condition is not None:
            return self.condition(state)
        return True

    def __repr__(self) -> str:
        return f"PipelineStep(agent={self.agent.name})"


class Pipeline:
    """Orchestrates sequential execution of pipeline steps.

    Supports conditional steps, retry logic, and an optimization loop
    that cycles through diagnosis -> optimization -> retraining -> re-evaluation.
    """

    def __init__(
        self,
        steps: list[PipelineStep],
        workflow_state: WorkflowState,
        message_bus: MessageBus,
    ) -> None:
        self.steps = steps
        self.state = workflow_state
        self.message_bus = message_bus

    async def run(self) -> WorkflowState:
        """Execute the full pipeline, returning final workflow state."""
        logger.info("pipeline_started", total_steps=len(self.steps))

        for step in self.steps:
            if not step.should_run(self.state):
                step.agent.status = AgentStatus.SKIPPED
                logger.info("step_skipped", agent=step.agent.name)
                continue

            retries = 0
            while True:
                try:
                    await step.agent.run()
                    break
                except Exception as e:
                    retries += 1
                    if step.retry_on_failure and retries <= step.max_retries:
                        logger.warning(
                            "step_retry",
                            agent=step.agent.name,
                            attempt=retries,
                            error=str(e),
                        )
                        continue
                    logger.error(
                        "step_failed",
                        agent=step.agent.name,
                        error=str(e),
                    )
                    raise

        logger.info("pipeline_completed")
        return self.state

    async def run_from(self, stage: PipelineStage) -> WorkflowState:
        """Resume execution from a specific stage."""
        started = False
        filtered_steps = []
        for step in self.steps:
            if step.agent.stage == stage:
                started = True
            if started:
                filtered_steps.append(step)

        original_steps = self.steps
        self.steps = filtered_steps
        try:
            return await self.run()
        finally:
            self.steps = original_steps


class PipelineBuilder:
    """Fluent API for constructing pipelines."""

    def __init__(self, workflow_state: WorkflowState, message_bus: MessageBus) -> None:
        self._steps: list[PipelineStep] = []
        self._state = workflow_state
        self._message_bus = message_bus

    def add_step(
        self,
        agent: BaseAgent,
        condition: Callable[[WorkflowState], bool] | None = None,
        retry_on_failure: bool = False,
        max_retries: int = 1,
    ) -> PipelineBuilder:
        """Add a pipeline step."""
        self._steps.append(
            PipelineStep(
                agent=agent,
                condition=condition,
                retry_on_failure=retry_on_failure,
                max_retries=max_retries,
            )
        )
        return self

    def add_optimization_loop(
        self,
        diagnosis_agent: BaseAgent,
        optimization_agent: BaseAgent,
        training_agent: BaseAgent,
        evaluation_agent: BaseAgent,
        max_rounds: int = 3,
    ) -> PipelineBuilder:
        """Add the diagnosis -> optimization -> retrain -> re-evaluate loop."""

        def needs_optimization(state: WorkflowState) -> bool:
            return state.needs_optimization

        # The loop agents only run if optimization is needed
        for agent in [optimization_agent, training_agent, evaluation_agent, diagnosis_agent]:
            self.add_step(agent, condition=needs_optimization, retry_on_failure=True)

        self._state.max_optimization_rounds = max_rounds
        return self

    def build(self) -> Pipeline:
        """Build and return the configured pipeline."""
        return Pipeline(
            steps=self._steps,
            workflow_state=self._state,
            message_bus=self._message_bus,
        )
