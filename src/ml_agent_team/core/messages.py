"""Agent message types and factory functions for inter-agent communication."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from uuid import UUID, uuid4

from .types import MessageType, PipelineStage


@dataclass(frozen=True, slots=True)
class AgentMessage:
    """Immutable message passed between agents via the message bus."""

    id: UUID = field(default_factory=uuid4)
    type: MessageType = MessageType.LOG
    source_agent: str = ""
    target_agent: str | None = None
    stage: PipelineStage | None = None
    payload: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    correlation_id: UUID | None = None


def command_message(
    source: str,
    target: str,
    stage: PipelineStage,
    params: dict[str, Any] | None = None,
    correlation_id: UUID | None = None,
) -> AgentMessage:
    """Create a command message from orchestrator to an agent."""
    return AgentMessage(
        type=MessageType.COMMAND,
        source_agent=source,
        target_agent=target,
        stage=stage,
        payload=params or {},
        correlation_id=correlation_id,
    )


def result_message(
    source: str,
    stage: PipelineStage,
    artifacts: dict[str, Any],
    correlation_id: UUID | None = None,
) -> AgentMessage:
    """Create a result message from an agent back to the orchestrator."""
    return AgentMessage(
        type=MessageType.RESULT,
        source_agent=source,
        stage=stage,
        payload=artifacts,
        correlation_id=correlation_id,
    )


def error_message(
    source: str,
    error: Exception,
    stage: PipelineStage | None = None,
    correlation_id: UUID | None = None,
) -> AgentMessage:
    """Create an error message."""
    return AgentMessage(
        type=MessageType.ERROR,
        source_agent=source,
        stage=stage,
        payload={"error_type": type(error).__name__, "error_message": str(error)},
        correlation_id=correlation_id,
    )


def review_request(
    source: str,
    stage: PipelineStage,
    work_summary: dict[str, Any],
) -> AgentMessage:
    """Create a review request for the peer review agent."""
    return AgentMessage(
        type=MessageType.REVIEW_REQUEST,
        source_agent=source,
        target_agent="peer_review",
        stage=stage,
        payload=work_summary,
    )


def review_result_message(
    findings: list[dict[str, Any]],
    approved: bool,
    reviewed_stage: PipelineStage,
    target: str,
) -> AgentMessage:
    """Create a review result from the peer review agent."""
    return AgentMessage(
        type=MessageType.REVIEW_RESULT,
        source_agent="peer_review",
        target_agent=target,
        stage=reviewed_stage,
        payload={"approved": approved, "findings": findings},
    )
