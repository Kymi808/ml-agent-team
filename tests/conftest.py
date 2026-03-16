"""Shared test fixtures."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ml_agent_team.core.config import AgentConfig
from ml_agent_team.core.message_bus import MessageBus
from ml_agent_team.core.workflow_state import WorkflowState


@pytest.fixture
def message_bus() -> MessageBus:
    """Create a fresh message bus."""
    return MessageBus()


@pytest.fixture
def workflow_state() -> WorkflowState:
    """Create a fresh workflow state."""
    return WorkflowState()


@pytest.fixture
def agent_config() -> AgentConfig:
    """Create a default agent config."""
    return AgentConfig()


@pytest.fixture
def sample_classification_df() -> pd.DataFrame:
    """Create a sample binary classification dataset."""
    rng = np.random.RandomState(42)
    n = 200
    return pd.DataFrame({
        "feature_1": rng.randn(n),
        "feature_2": rng.randn(n),
        "feature_3": rng.rand(n),
        "category": rng.choice(["A", "B", "C"], n),
        "target": rng.choice([0, 1], n),
    })


@pytest.fixture
def sample_regression_df() -> pd.DataFrame:
    """Create a sample regression dataset."""
    rng = np.random.RandomState(42)
    n = 200
    x1 = rng.randn(n)
    x2 = rng.randn(n)
    return pd.DataFrame({
        "feature_1": x1,
        "feature_2": x2,
        "feature_3": rng.rand(n),
        "target": 3 * x1 + 2 * x2 + rng.randn(n) * 0.5,
    })


@pytest.fixture
def sample_csv_path(tmp_path: object, sample_classification_df: pd.DataFrame) -> str:
    """Write sample classification data to a CSV file and return the path."""
    path = str(tmp_path / "sample.csv")  # type: ignore[operator]
    sample_classification_df.to_csv(path, index=False)
    return path
