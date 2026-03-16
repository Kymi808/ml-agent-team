"""Tests for configuration loading."""

import yaml
import pytest

from ml_agent_team.core.config import AgentConfig, PipelineConfig, load_config


def test_default_pipeline_config():
    config = PipelineConfig()
    assert config.project_name == "ml_experiment"
    assert config.data.test_size == 0.2
    assert config.training.cross_validation_folds == 5
    assert config.enable_peer_review is True


def test_get_agent_config_default():
    config = PipelineConfig()
    agent_cfg = config.get_agent_config("nonexistent_agent")
    assert agent_cfg.enabled is True
    assert agent_cfg.timeout_seconds == 300


def test_get_agent_config_custom():
    config = PipelineConfig(
        agents={"training": AgentConfig(timeout_seconds=900)}
    )
    agent_cfg = config.get_agent_config("training")
    assert agent_cfg.timeout_seconds == 900


def test_load_config_from_yaml(tmp_path):
    config_data = {
        "project_name": "test_project",
        "max_optimization_rounds": 5,
        "data": {"test_size": 0.3},
    }
    path = tmp_path / "config.yaml"
    with open(path, "w") as f:
        yaml.dump(config_data, f)

    config = load_config(path)
    assert config.project_name == "test_project"
    assert config.max_optimization_rounds == 5
    assert config.data.test_size == 0.3
    # Defaults should still be present
    assert config.enable_peer_review is True
