"""Tests for the message system."""

from ml_agent_team.core.messages import (
    AgentMessage,
    command_message,
    error_message,
    result_message,
    review_request,
)
from ml_agent_team.core.types import MessageType, PipelineStage


def test_agent_message_creation():
    msg = AgentMessage(source_agent="test", type=MessageType.LOG)
    assert msg.source_agent == "test"
    assert msg.type == MessageType.LOG
    assert msg.id is not None
    assert msg.timestamp is not None


def test_command_message():
    msg = command_message("orchestrator", "eda", PipelineStage.EDA, {"key": "value"})
    assert msg.type == MessageType.COMMAND
    assert msg.source_agent == "orchestrator"
    assert msg.target_agent == "eda"
    assert msg.stage == PipelineStage.EDA
    assert msg.payload == {"key": "value"}


def test_result_message():
    msg = result_message("eda", PipelineStage.EDA, {"plots": 3})
    assert msg.type == MessageType.RESULT
    assert msg.payload == {"plots": 3}


def test_error_message():
    msg = error_message("training", ValueError("bad input"), PipelineStage.TRAINING)
    assert msg.type == MessageType.ERROR
    assert "ValueError" in msg.payload["error_type"]
    assert "bad input" in msg.payload["error_message"]


def test_review_request():
    msg = review_request("evaluation", PipelineStage.EVALUATION, {"metrics": {"f1": 0.9}})
    assert msg.type == MessageType.REVIEW_REQUEST
    assert msg.target_agent == "peer_review"


def test_message_immutability():
    msg = AgentMessage()
    try:
        msg.source_agent = "changed"  # type: ignore[misc]
        assert False, "Should not allow mutation"
    except AttributeError:
        pass
