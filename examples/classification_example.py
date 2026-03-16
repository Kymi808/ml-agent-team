"""Example: Run the ML Agent Team on a classification problem.

Usage:
    python examples/classification_example.py
"""

import asyncio
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from ml_agent_team import OrchestratorAgent, PipelineConfig


def create_sample_dataset() -> str:
    """Create a synthetic classification dataset."""
    rng = np.random.RandomState(42)
    n = 500

    # Generate features
    age = rng.randint(18, 80, n)
    income = rng.normal(50000, 20000, n).clip(15000)
    usage_hours = rng.exponential(5, n)
    tenure_months = rng.randint(1, 120, n)
    support_calls = rng.poisson(2, n)
    plan = rng.choice(["basic", "premium", "enterprise"], n, p=[0.5, 0.35, 0.15])

    # Generate target (churn) with some logic
    churn_prob = (
        0.1
        + 0.3 * (tenure_months < 12).astype(float)
        + 0.2 * (support_calls > 3).astype(float)
        - 0.15 * (plan == "enterprise").astype(float)
        + 0.1 * (usage_hours < 2).astype(float)
    )
    churn_prob = churn_prob.clip(0.05, 0.95)
    churned = rng.binomial(1, churn_prob)

    # Add some missing values
    income_with_missing = income.copy().astype(float)
    income_with_missing[rng.choice(n, 25, replace=False)] = np.nan

    df = pd.DataFrame({
        "age": age,
        "income": income_with_missing,
        "usage_hours_per_week": usage_hours.round(1),
        "tenure_months": tenure_months,
        "support_calls": support_calls,
        "plan": plan,
        "churned": churned,
    })

    # Save to temp file
    path = Path(tempfile.mkdtemp()) / "churn_data.csv"
    df.to_csv(path, index=False)
    print(f"Dataset created: {path}")
    print(f"  Samples: {n}")
    print(f"  Churn rate: {churned.mean():.1%}")
    return str(path)


async def main() -> None:
    # Create dataset
    data_path = create_sample_dataset()

    # Configure pipeline
    config = PipelineConfig(
        project_name="churn_prediction",
        output_dir="./output/churn_example",
        max_optimization_rounds=2,
        data={"target_column": "churned", "test_size": 0.2},
        training={
            "cross_validation_folds": 5,
            "hyperparameter_tuning": False,  # Set True for better results (slower)
        },
    )

    # Run pipeline
    orchestrator = OrchestratorAgent(config)
    result = await orchestrator.run(
        problem_description=(
            "Predict customer churn based on demographics, usage patterns, "
            "and support interaction history"
        ),
        data_source=data_path,
    )

    # Print results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Problem Type: {result.problem.problem_type}")
    print(f"Best Model: {result.best_model_name}")
    print(f"Acceptable: {result.is_acceptable}")
    print(f"\nMetrics:")
    for metric, value in result.metrics.items():
        print(f"  {metric}: {value:.4f}")
    if result.report_path:
        print(f"\nFull report: {result.report_path}")


if __name__ == "__main__":
    asyncio.run(main())
