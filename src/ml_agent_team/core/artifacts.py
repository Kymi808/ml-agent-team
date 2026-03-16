"""Artifact registry for tracking and persisting pipeline outputs."""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .types import PipelineStage


@dataclass(slots=True)
class Artifact:
    """A named artifact produced by an agent during pipeline execution."""

    name: str
    artifact_type: str  # "dataset", "model", "plot", "report", "config"
    stage: PipelineStage
    created_by: str
    path: Path | None = None
    value: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class ArtifactRegistry:
    """Central registry for all pipeline artifacts with optional disk persistence."""

    def __init__(self) -> None:
        self._artifacts: dict[str, Artifact] = {}

    def register(self, artifact: Artifact) -> None:
        """Register an artifact."""
        self._artifacts[artifact.name] = artifact

    def get(self, name: str) -> Artifact | None:
        """Get an artifact by name."""
        return self._artifacts.get(name)

    def list_by_stage(self, stage: PipelineStage) -> list[Artifact]:
        """List all artifacts from a given pipeline stage."""
        return [a for a in self._artifacts.values() if a.stage == stage]

    def list_by_type(self, artifact_type: str) -> list[Artifact]:
        """List all artifacts of a given type."""
        return [a for a in self._artifacts.values() if a.artifact_type == artifact_type]

    def list_all(self) -> list[Artifact]:
        """List all registered artifacts."""
        return list(self._artifacts.values())

    def persist(self, name: str, directory: Path) -> Path:
        """Persist a single artifact to disk."""
        artifact = self._artifacts.get(name)
        if artifact is None:
            raise KeyError(f"Artifact '{name}' not found")

        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{name}.pkl"

        with open(path, "wb") as f:
            pickle.dump(artifact.value, f)

        artifact.path = path
        return path

    def persist_all(self, directory: Path) -> dict[str, Path]:
        """Persist all artifacts to disk."""
        paths: dict[str, Path] = {}
        for name in self._artifacts:
            try:
                paths[name] = self.persist(name, directory)
            except Exception:
                continue
        return paths

    @property
    def count(self) -> int:
        """Number of registered artifacts."""
        return len(self._artifacts)
