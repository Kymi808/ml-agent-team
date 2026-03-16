"""Tests for workflow state."""

from ml_agent_team.core.types import PipelineStage
from ml_agent_team.core.workflow_state import WorkflowState


def test_mark_stage_completed():
    state = WorkflowState()
    state.mark_stage_completed(PipelineStage.EDA)

    assert state.is_stage_completed(PipelineStage.EDA)
    assert not state.is_stage_completed(PipelineStage.TRAINING)


def test_mark_stage_failed_then_completed():
    state = WorkflowState()
    state.mark_stage_failed(PipelineStage.TRAINING)
    assert PipelineStage.TRAINING in state.failed_stages

    state.mark_stage_completed(PipelineStage.TRAINING)
    assert PipelineStage.TRAINING not in state.failed_stages
    assert state.is_stage_completed(PipelineStage.TRAINING)


def test_needs_optimization():
    state = WorkflowState()
    state.is_acceptable = False
    state.issues = [{"type": "overfitting"}]
    state.max_optimization_rounds = 3
    state.optimization_rounds = 0

    assert state.needs_optimization

    state.optimization_rounds = 3
    assert not state.needs_optimization


def test_artifact_store():
    state = WorkflowState()
    state.set_artifact("my_data", [1, 2, 3])
    assert state.get_artifact("my_data") == [1, 2, 3]
    assert state.get_artifact("missing") is None


def test_no_duplicate_completed_stages():
    state = WorkflowState()
    state.mark_stage_completed(PipelineStage.EDA)
    state.mark_stage_completed(PipelineStage.EDA)
    assert state.completed_stages.count(PipelineStage.EDA) == 1
