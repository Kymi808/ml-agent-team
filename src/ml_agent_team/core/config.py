"""Configuration models and YAML loader for the pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


class AgentConfig(BaseModel):
    """Configuration for a single agent."""

    enabled: bool = True
    timeout_seconds: int = 300
    retry_on_failure: bool = False
    max_retries: int = 1
    params: dict[str, Any] = Field(default_factory=dict)


class DataConfig(BaseModel):
    """Data source configuration."""

    source_path: str | None = None
    source_url: str | None = None
    file_format: str = "auto"
    target_column: str | None = None
    test_size: float = 0.2
    random_state: int = 42
    stratify: bool = True


class TrainingConfig(BaseModel):
    """Training and tuning configuration."""

    cross_validation_folds: int = 5
    early_stopping: bool = True
    early_stopping_patience: int = 10
    hyperparameter_tuning: bool = True
    tuning_method: str = "optuna"  # optuna, grid, random
    tuning_trials: int = 50
    scoring_metric: str | None = None  # auto-detected if None


class LLMConfig(BaseModel):
    """LLM integration configuration for agents that use LLM reasoning."""

    provider: str = "anthropic"
    model: str = "claude-sonnet-4-20250514"
    api_key_env: str = "ANTHROPIC_API_KEY"
    temperature: float = 0.2
    max_tokens: int = 4096


class PipelineConfig(BaseModel):
    """Top-level pipeline configuration."""

    project_name: str = "ml_experiment"
    output_dir: str = "./output"
    log_level: str = "INFO"
    max_optimization_rounds: int = 3
    enable_peer_review: bool = True
    enable_literature_review: bool = True
    data: DataConfig = Field(default_factory=DataConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    agents: dict[str, AgentConfig] = Field(default_factory=dict)

    def get_agent_config(self, agent_name: str) -> AgentConfig:
        """Get config for a specific agent, falling back to defaults."""
        return self.agents.get(agent_name, AgentConfig())


def load_config(config_path: Path) -> PipelineConfig:
    """Load pipeline configuration from a YAML file, merging with defaults."""
    with open(config_path) as f:
        raw = yaml.safe_load(f) or {}

    return PipelineConfig(**raw)


def save_config(config: PipelineConfig, config_path: Path) -> None:
    """Save pipeline configuration to a YAML file."""
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w") as f:
        yaml.dump(config.model_dump(), f, default_flow_style=False, sort_keys=False)
