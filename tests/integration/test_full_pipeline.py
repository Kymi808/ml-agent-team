"""Integration test: run the full pipeline on a toy dataset."""

import pytest
import numpy as np
import pandas as pd

from ml_agent_team.core.config import PipelineConfig
from ml_agent_team.agents.orchestrator import OrchestratorAgent


@pytest.fixture
def toy_dataset(tmp_path) -> str:
    """Create a simple classification dataset CSV."""
    rng = np.random.RandomState(42)
    n = 200
    x1 = rng.randn(n)
    x2 = rng.randn(n)
    target = (x1 + x2 > 0).astype(int)
    # Add some noise
    flip_idx = rng.choice(n, size=20, replace=False)
    target[flip_idx] = 1 - target[flip_idx]

    df = pd.DataFrame({
        "feature_1": x1,
        "feature_2": x2,
        "feature_3": rng.rand(n),
        "category_a": rng.choice(["low", "medium", "high"], n),
        "target": target,
    })

    path = str(tmp_path / "toy_data.csv")
    df.to_csv(path, index=False)
    return path


@pytest.mark.asyncio
async def test_full_classification_pipeline(toy_dataset, tmp_path):
    """Run the complete pipeline end-to-end on a toy classification dataset."""
    config = PipelineConfig(
        project_name="test_classification",
        output_dir=str(tmp_path / "output"),
        max_optimization_rounds=1,
        enable_peer_review=True,
        enable_literature_review=True,
        data={"target_column": "target", "test_size": 0.2, "random_state": 42},
        training={
            "cross_validation_folds": 3,
            "hyperparameter_tuning": False,  # Skip tuning for speed
        },
    )

    orchestrator = OrchestratorAgent(config)
    result = await orchestrator.run(
        problem_description="Classify samples as positive or negative based on features",
        data_source=toy_dataset,
    )

    # Verify all key stages completed
    assert result.problem.problem_type is not None
    assert result.raw_data is not None
    assert len(result.eda_report) > 0
    assert result.X_train is not None
    assert result.X_test is not None
    assert result.trained_model is not None
    assert len(result.metrics) > 0
    assert result.report_path is not None

    # Verify metrics are reasonable (not random chance)
    assert result.metrics.get("accuracy", 0) > 0.5
