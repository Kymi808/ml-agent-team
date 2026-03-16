"""Example: Run the ML Agent Team on a regression problem.

Usage:
    python examples/regression_example.py
"""

import asyncio
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from ml_agent_team import OrchestratorAgent, PipelineConfig


def create_housing_dataset() -> str:
    """Create a synthetic housing price dataset."""
    rng = np.random.RandomState(42)
    n = 400

    sqft = rng.normal(1800, 500, n).clip(500)
    bedrooms = rng.choice([1, 2, 3, 4, 5], n, p=[0.05, 0.2, 0.4, 0.25, 0.1])
    age = rng.randint(0, 80, n)
    neighborhood = rng.choice(["downtown", "suburbs", "rural"], n, p=[0.3, 0.5, 0.2])

    # Price formula
    price = (
        50000
        + 150 * sqft
        + 20000 * bedrooms
        - 500 * age
        + 30000 * (neighborhood == "downtown").astype(float)
        + rng.normal(0, 15000, n)
    ).clip(50000)

    df = pd.DataFrame({
        "sqft": sqft.round(0).astype(int),
        "bedrooms": bedrooms,
        "age_years": age,
        "neighborhood": neighborhood,
        "lot_size_acres": (rng.exponential(0.3, n) + 0.1).round(2),
        "price": price.round(-2),
    })

    path = Path(tempfile.mkdtemp()) / "housing_data.csv"
    df.to_csv(path, index=False)
    print(f"Dataset created: {path} ({n} samples)")
    return str(path)


async def main() -> None:
    data_path = create_housing_dataset()

    config = PipelineConfig(
        project_name="house_price_prediction",
        output_dir="./output/housing_example",
        data={"target_column": "price"},
        training={"hyperparameter_tuning": False},
    )

    orchestrator = OrchestratorAgent(config)
    result = await orchestrator.run(
        problem_description="Predict house prices based on property characteristics",
        data_source=data_path,
    )

    print(f"\nBest Model: {result.best_model_name}")
    print(f"Metrics: {result.metrics}")
    if result.report_path:
        print(f"Report: {result.report_path}")


if __name__ == "__main__":
    asyncio.run(main())
