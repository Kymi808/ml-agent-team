"""Tests for the pipeline engine."""

import pytest

from ml_agent_team.core.base_agent import BaseAgent
from ml_agent_team.core.config import AgentConfig
from ml_agent_team.core.message_bus import MessageBus
from ml_agent_team.core.messages import AgentMessage, result_message
from ml_agent_team.core.pipeline import Pipeline, PipelineBuilder, PipelineStep
from ml_agent_team.core.types import PipelineStage
from ml_agent_team.core.workflow_state import WorkflowState


class DummyAgent(BaseAgent):
    """A simple agent for testing."""

    def __init__(self, name: str, stage_val: PipelineStage, **kwargs):
        self._stage = stage_val
        super().__init__(name=name, **kwargs)

    @property
    def stage(self) -> PipelineStage:
        return self._stage

    @property
    def description(self) -> str:
        return "Dummy agent for testing"

    async def execute(self) -> AgentMessage:
        self.state.set_artifact(f"{self.name}_ran", True)
        return result_message(self.name, self.stage, {"status": "ok"})


@pytest.fixture
def setup():
    bus = MessageBus()
    state = WorkflowState()
    config = AgentConfig()
    return bus, state, config


@pytest.mark.asyncio
async def test_pipeline_runs_all_steps(setup):
    bus, state, config = setup
    a1 = DummyAgent("a1", PipelineStage.EDA, config=config, message_bus=bus, workflow_state=state)
    a2 = DummyAgent("a2", PipelineStage.TRAINING, config=config, message_bus=bus, workflow_state=state)

    pipeline = Pipeline(
        steps=[PipelineStep(a1), PipelineStep(a2)],
        workflow_state=state,
        message_bus=bus,
    )

    result = await pipeline.run()
    assert result.get_artifact("a1_ran") is True
    assert result.get_artifact("a2_ran") is True


@pytest.mark.asyncio
async def test_pipeline_skips_conditional_step(setup):
    bus, state, config = setup
    a1 = DummyAgent("a1", PipelineStage.EDA, config=config, message_bus=bus, workflow_state=state)

    pipeline = Pipeline(
        steps=[PipelineStep(a1, condition=lambda s: False)],
        workflow_state=state,
        message_bus=bus,
    )

    result = await pipeline.run()
    assert result.get_artifact("a1_ran") is None


@pytest.mark.asyncio
async def test_pipeline_builder(setup):
    bus, state, config = setup
    a1 = DummyAgent("a1", PipelineStage.EDA, config=config, message_bus=bus, workflow_state=state)
    a2 = DummyAgent("a2", PipelineStage.TRAINING, config=config, message_bus=bus, workflow_state=state)

    pipeline = PipelineBuilder(state, bus).add_step(a1).add_step(a2).build()
    result = await pipeline.run()
    assert result.get_artifact("a1_ran") is True
    assert result.get_artifact("a2_ran") is True
