"""Shared type definitions, enums, and type aliases used across the package."""

from __future__ import annotations

import sys
from enum import Enum, auto
from typing import Any, TypeAlias

if sys.version_info >= (3, 11):
    from enum import StrEnum
else:

    class StrEnum(str, Enum):  # type: ignore[no-redef]
        """Backport of StrEnum for Python 3.10."""

        @staticmethod
        def _generate_next_value_(
            name: str,
            start: int,
            count: int,
            last_values: list,  # noqa: ARG004
        ) -> str:
            return name.lower()


class ProblemType(StrEnum):
    """Types of ML problems the system can handle."""

    BINARY_CLASSIFICATION = auto()
    MULTICLASS_CLASSIFICATION = auto()
    REGRESSION = auto()
    CLUSTERING = auto()
    TIME_SERIES = auto()
    ANOMALY_DETECTION = auto()
    RANKING = auto()
    RECOMMENDATION = auto()


class AgentStatus(StrEnum):
    """Execution status of an agent."""

    IDLE = auto()
    RUNNING = auto()
    COMPLETED = auto()
    FAILED = auto()
    SKIPPED = auto()
    WAITING_FOR_REVIEW = auto()


class PipelineStage(StrEnum):
    """Ordered stages in the ML pipeline."""

    PROBLEM_ANALYSIS = auto()
    LITERATURE_REVIEW = auto()
    DATA_INGESTION = auto()
    EDA = auto()
    FEATURE_ENGINEERING = auto()
    MODEL_SELECTION = auto()
    MODEL_BUILDING = auto()
    TRAINING = auto()
    EVALUATION = auto()
    DIAGNOSIS = auto()
    OPTIMIZATION = auto()
    REPORTING = auto()


class MessageType(StrEnum):
    """Types of messages exchanged between agents."""

    COMMAND = auto()
    RESULT = auto()
    STATUS_UPDATE = auto()
    REVIEW_REQUEST = auto()
    REVIEW_RESULT = auto()
    ERROR = auto()
    LOG = auto()


class Severity(StrEnum):
    """Severity levels for issues found during diagnosis or review."""

    INFO = auto()
    WARNING = auto()
    ERROR = auto()
    CRITICAL = auto()


# Type aliases for flexibility across frameworks
DataFrameLike: TypeAlias = Any
ModelLike: TypeAlias = Any
