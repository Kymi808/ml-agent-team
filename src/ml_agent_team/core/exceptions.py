"""Custom exception hierarchy for the ML Agent Team package."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .types import PipelineStage


class MLAgentTeamError(Exception):
    """Base exception for the package."""


class AgentExecutionError(MLAgentTeamError):
    """Raised when an agent fails during execution."""

    def __init__(self, agent_name: str, stage: PipelineStage, cause: Exception) -> None:
        self.agent_name = agent_name
        self.stage = stage
        self.cause = cause
        super().__init__(f"Agent '{agent_name}' failed at stage '{stage}': {cause}")


class PipelineError(MLAgentTeamError):
    """Raised when the pipeline encounters an unrecoverable error."""


class ConfigurationError(MLAgentTeamError):
    """Raised for invalid configuration."""


class DataValidationError(MLAgentTeamError):
    """Raised when data fails validation checks."""


class ReviewRejectionError(MLAgentTeamError):
    """Raised when peer review rejects an agent's work, triggering a retry."""

    def __init__(self, agent_name: str, findings: list[dict[str, Any]]) -> None:
        self.agent_name = agent_name
        self.findings = findings
        super().__init__(
            f"Peer review rejected work from '{agent_name}': "
            f"{len(findings)} issue(s) found"
        )
