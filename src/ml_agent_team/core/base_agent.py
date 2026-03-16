"""Base agent abstract class defining the lifecycle contract for all agents."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import structlog

from .config import AgentConfig
from .exceptions import AgentExecutionError
from .message_bus import MessageBus
from .messages import AgentMessage, error_message, result_message, review_request
from .types import AgentStatus, PipelineStage
from .workflow_state import WorkflowState


class BaseAgent(ABC):
    """Abstract base class for all agents in the ML pipeline.

    Defines the lifecycle: initialize -> execute -> cleanup.
    Each agent reads from and writes to the shared WorkflowState,
    and communicates coordination signals via the MessageBus.
    """

    def __init__(
        self,
        name: str,
        config: AgentConfig,
        message_bus: MessageBus,
        workflow_state: WorkflowState,
    ) -> None:
        self.name = name
        self.config = config
        self.message_bus = message_bus
        self.state = workflow_state
        self.status: AgentStatus = AgentStatus.IDLE
        self.logger = structlog.get_logger(self.name)

        # Register with message bus to receive targeted messages
        self.message_bus.subscribe(self.name, self._handle_message)

    @property
    @abstractmethod
    def stage(self) -> PipelineStage:
        """Which pipeline stage this agent belongs to."""
        ...

    @property
    @abstractmethod
    def description(self) -> str:
        """Human-readable description of this agent's role."""
        ...

    @property
    def dependencies(self) -> list[PipelineStage]:
        """Stages that must complete before this agent can run. Override as needed."""
        return []

    async def initialize(self) -> None:
        """Optional hook: prepare resources before execution."""

    @abstractmethod
    async def execute(self) -> AgentMessage:
        """Main execution logic.

        Reads from self.state, writes results back to self.state,
        and returns a summary AgentMessage.
        """
        ...

    async def cleanup(self) -> None:
        """Optional hook: release resources after execution."""

    async def run(self) -> AgentMessage:
        """Full lifecycle: initialize -> execute -> cleanup with error handling."""
        self.status = AgentStatus.RUNNING
        self.state.current_stage = self.stage
        self.logger.info("agent_started", stage=self.stage)

        try:
            await self.initialize()
            result = await self.execute()
            self.status = AgentStatus.COMPLETED
            self.state.mark_stage_completed(self.stage)
            await self.message_bus.publish(result)
            self.logger.info("agent_completed", stage=self.stage)
            return result
        except Exception as e:
            self.status = AgentStatus.FAILED
            self.state.mark_stage_failed(self.stage)
            err_msg = error_message(self.name, e, self.stage)
            await self.message_bus.publish(err_msg)
            self.logger.error("agent_failed", stage=self.stage, error=str(e))
            raise AgentExecutionError(self.name, self.stage, e) from e
        finally:
            await self.cleanup()

    async def _handle_message(self, message: AgentMessage) -> None:
        """Handle incoming messages. Override for reactive behavior."""

    async def request_peer_review(self, work_summary: dict[str, Any]) -> None:
        """Ask the PeerReviewAgent to review this agent's work."""
        msg = review_request(self.name, self.stage, work_summary)
        await self.message_bus.publish(msg)

    def _result_message(self, artifacts: dict[str, Any]) -> AgentMessage:
        """Convenience method to create a result message from this agent."""
        return result_message(self.name, self.stage, artifacts)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r}, status={self.status})"
