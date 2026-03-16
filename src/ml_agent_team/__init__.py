"""ML Agent Team — a multi-agent system for end-to-end ML workflows."""

from .core.base_agent import BaseAgent
from .core.config import PipelineConfig, load_config
from .core.message_bus import MessageBus
from .core.messages import AgentMessage
from .core.pipeline import Pipeline, PipelineBuilder
from .core.types import AgentStatus, MessageType, PipelineStage, ProblemType
from .core.workflow_state import WorkflowState
from .agents.orchestrator import OrchestratorAgent

__version__ = "0.1.0"

__all__ = [
    "BaseAgent",
    "AgentMessage",
    "AgentStatus",
    "MessageBus",
    "MessageType",
    "OrchestratorAgent",
    "Pipeline",
    "PipelineBuilder",
    "PipelineConfig",
    "PipelineStage",
    "ProblemType",
    "WorkflowState",
    "load_config",
]
