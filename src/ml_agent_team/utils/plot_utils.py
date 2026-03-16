"""Plotting utility functions for consistent, publication-quality visualizations."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# Default style settings
STYLE_CONFIG = {
    "figure.figsize": (10, 6),
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 150,
}

COLORS = {
    "primary": "#4C72B0",
    "secondary": "#DD8452",
    "success": "#55A868",
    "danger": "#C44E52",
    "neutral": "#8C8C8C",
    "palette": ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3", "#937860"],
}


def apply_style() -> None:
    """Apply consistent matplotlib style."""
    plt.rcParams.update(STYLE_CONFIG)


def save_figure(fig: plt.Figure, path: str | Path, close: bool = True) -> str:
    """Save a matplotlib figure and optionally close it."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(path), dpi=150, bbox_inches="tight")
    if close:
        plt.close(fig)
    return str(path)


def bar_chart(
    labels: list[str],
    values: list[float],
    title: str = "",
    xlabel: str = "",
    ylabel: str = "",
    color: str = COLORS["primary"],
    highlight_max: bool = False,
) -> plt.Figure:
    """Create a bar chart."""
    apply_style()
    fig, ax = plt.subplots()
    bars = ax.bar(labels, values, color=color, edgecolor="white")

    if highlight_max and values:
        max_idx = int(np.argmax(values))
        bars[max_idx].set_color(COLORS["success"])

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    return fig
