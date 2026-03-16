# ML Agent Team

A multi-agent system that orchestrates a full end-to-end machine learning workflow — from problem analysis through model training, evaluation, optimization, and comprehensive reporting.

## Overview

ML Agent Team automates the complete ML lifecycle using specialized, collaborating agents. Each agent handles a distinct phase of the workflow, communicating through a shared state blackboard and async message bus. The system mirrors how a team of ML engineers would tackle a problem: analyze, explore, build, evaluate, iterate, and report.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Orchestrator Agent                           │
│  Coordinates pipeline execution, manages failures, routes messages  │
└─────────┬───────────────────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────────────────────┐
│                          Pipeline Engine                             │
│  Sequential execution · Conditional steps · Optimization loops      │
└─────────┬───────────────────────────────────────────────────────────┘
          │
    ┌─────┴──────────────────────────────────────────────┐
    │              Agent Execution Flow                    │
    │                                                     │
    │  ┌──────────────┐    ┌───────────────────┐         │
    │  │   Problem     │───▶│ Literature Review │         │
    │  │   Analysis    │    │   (optional)      │         │
    │  └──────┬───────┘    └───────┬───────────┘         │
    │         │                    │                       │
    │         ▼                    ▼                       │
    │  ┌──────────────┐    ┌───────────────────┐         │
    │  │    Data       │───▶│   Exploratory     │         │
    │  │  Ingestion    │    │  Data Analysis    │         │
    │  └──────────────┘    └───────┬───────────┘         │
    │                              │                       │
    │                              ▼                       │
    │                      ┌───────────────────┐          │
    │                      │     Feature        │          │
    │                      │   Engineering      │          │
    │                      └───────┬───────────┘          │
    │                              │                       │
    │                              ▼                       │
    │  ┌──────────────┐    ┌───────────────────┐         │
    │  │    Model      │───▶│     Model         │         │
    │  │  Selection    │    │    Building       │         │
    │  └──────────────┘    └───────┬───────────┘         │
    │                              │                       │
    │                              ▼                       │
    │                      ┌───────────────────┐          │
    │               ┌─────▶│    Training &     │          │
    │               │      │  Hyperparameter   │          │
    │               │      │     Tuning        │          │
    │               │      └───────┬───────────┘          │
    │               │              │                       │
    │               │              ▼                       │
    │               │      ┌───────────────────┐          │
    │               │      │   Evaluation       │         │
    │               │      └───────┬───────────┘          │
    │               │              │                       │
    │               │              ▼                       │
    │               │      ┌───────────────────┐          │
    │               │      │   Diagnosis &      │         │
    │               │      │  Error Analysis    │         │
    │               │      └───────┬───────────┘          │
    │               │              │                       │
    │               │         Issues found?                │
    │               │        ╱            ╲                │
    │               │      Yes             No              │
    │               │      ╱                 ╲             │
    │        ┌──────┴──────┐          ┌──────────────┐   │
    │        │ Optimization │          │  Reporting &  │   │
    │        │   & Retry    │          │  Conclusions  │   │
    │        └─────────────┘          └──────────────┘   │
    │                                                     │
    │  ┌──────────────────────────────────────────────┐  │
    │  │  Peer Review Agent (cross-cutting validator)  │  │
    │  │  Reviews methodology at configurable gates    │  │
    │  └──────────────────────────────────────────────┘  │
    └─────────────────────────────────────────────────────┘
```

## Agents

| Agent | Role |
|-------|------|
| **Orchestrator** | Coordinates the entire pipeline, manages agent lifecycle and failure recovery |
| **Problem Analysis** | Parses the problem description, identifies ML task type, defines objectives and success criteria |
| **Literature Review** | Researches relevant approaches and state-of-the-art methods for the problem type |
| **Data Ingestion** | Loads and validates data from various sources (CSV, Parquet, JSON, etc.) |
| **EDA** | Performs extensive exploratory data analysis — distributions, correlations, outliers, missing patterns |
| **Feature Engineering** | Handles imputation, encoding, scaling, feature creation, and train/test splitting |
| **Model Selection** | Recommends candidate models based on problem type, data characteristics, and literature |
| **Model Building** | Constructs model pipelines with preprocessing and hyperparameter search spaces |
| **Training** | Trains models with cross-validation, hyperparameter tuning, and early stopping |
| **Evaluation** | Selects appropriate metrics, evaluates on test set, generates performance visualizations |
| **Diagnosis** | Identifies model weaknesses — overfitting, bias, calibration issues, error patterns |
| **Optimization** | Applies targeted fixes for diagnosed issues and triggers retraining when needed |
| **Reporting** | Generates comprehensive reports with visualizations, conclusions, and recommendations |
| **Peer Review** | Validates methodology, checks for data leakage, statistical validity, and reproducibility |

## Installation

```bash
pip install ml-agent-team
```

With optional dependencies:

```bash
# LLM-powered agents (problem analysis, literature review, reporting)
pip install ml-agent-team[llm]

# Gradient boosting models
pip install ml-agent-team[xgboost,lightgbm]

# Hyperparameter tuning with Optuna
pip install ml-agent-team[optuna]

# Experiment tracking
pip install ml-agent-team[tracking]

# Everything
pip install ml-agent-team[all]
```

For development:

```bash
git clone https://github.com/Kymi808/ml-agent-team.git
cd ml-agent-team
pip install -e ".[dev,all]"
```

## Quick Start

### CLI

```bash
# Run full pipeline on a dataset
ml-agent-team run "Predict customer churn from usage data" ./data/churn.csv

# With custom config
ml-agent-team run "Classify images" ./data/images/ --config configs/custom.yaml

# Skip optional agents
ml-agent-team run "Predict house prices" ./data/houses.csv --no-literature --no-review

# Resume from a saved state
ml-agent-team resume ./output/workflow_state.pkl --from-stage evaluation
```

### Python API

```python
import asyncio
from ml_agent_team import OrchestratorAgent, PipelineConfig

async def main():
    config = PipelineConfig(
        project_name="churn_prediction",
        data={"source_path": "./data/churn.csv", "target_column": "churned"},
        training={"hyperparameter_tuning": True, "tuning_trials": 100},
    )

    orchestrator = OrchestratorAgent(config)
    result = await orchestrator.run(
        problem_description="Predict which customers will churn based on usage patterns",
        data_source="./data/churn.csv",
    )

    print(f"Best model: {result.trained_model}")
    print(f"Metrics: {result.metrics}")
    print(f"Report: {result.report_path}")

asyncio.run(main())
```

### Custom Agents

```python
from ml_agent_team.core import BaseAgent, AgentMessage, PipelineStage

class DataAugmentationAgent(BaseAgent):
    """Custom agent that augments training data."""

    @property
    def stage(self) -> PipelineStage:
        return PipelineStage.FEATURE_ENGINEERING

    @property
    def description(self) -> str:
        return "Applies data augmentation techniques to increase training set diversity"

    @property
    def dependencies(self) -> list[PipelineStage]:
        return [PipelineStage.FEATURE_ENGINEERING]

    async def execute(self) -> AgentMessage:
        # Access shared state
        X_train = self.state.X_train
        y_train = self.state.y_train

        # Your augmentation logic here
        X_augmented, y_augmented = self.augment(X_train, y_train)

        # Write back to shared state
        self.state.X_train = X_augmented
        self.state.y_train = y_augmented

        return self.result_message({"augmented_samples": len(X_augmented)})
```

## Configuration

Configuration is done via YAML files. See `configs/default.yaml` for the full reference.

```yaml
project_name: my_experiment
output_dir: ./output
max_optimization_rounds: 3
enable_peer_review: true
enable_literature_review: true

data:
  source_path: ./data/dataset.csv
  target_column: target
  test_size: 0.2
  random_state: 42

training:
  cross_validation_folds: 5
  early_stopping: true
  hyperparameter_tuning: true
  tuning_method: optuna
  tuning_trials: 50

agents:
  training:
    timeout_seconds: 600
  literature_review:
    enabled: false
```

## Project Structure

```
ml_agent_team/
├── src/ml_agent_team/
│   ├── core/           # Core abstractions (BaseAgent, Pipeline, MessageBus, etc.)
│   ├── agents/         # All 14 agent implementations
│   ├── utils/          # Shared utilities (data, plotting, metrics, I/O)
│   └── integrations/   # Optional external integrations (LLM, MLflow, W&B)
├── configs/            # Default and per-agent configuration files
├── tests/              # Unit and integration tests
├── examples/           # Usage examples
└── docs/               # Documentation
```

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Run with coverage
pytest --cov=ml_agent_team

# Lint
ruff check src/

# Type check
mypy src/
```

## License

MIT
