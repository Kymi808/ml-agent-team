"""Command-line interface for the ML Agent Team pipeline."""

from __future__ import annotations

import asyncio
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from .core.config import PipelineConfig, load_config
from .core.logging import setup_logging

app = typer.Typer(
    name="ml-agent-team",
    help="Multi-agent system for end-to-end ML workflows",
    add_completion=False,
)
console = Console()


@app.command()
def run(
    problem: str = typer.Argument(..., help="Problem description in natural language"),
    data: Path = typer.Argument(..., help="Path to the data file"),
    config: Path | None = typer.Option(None, "--config", "-c", help="Path to config YAML"),
    output: str = typer.Option("./output", "--output", "-o", help="Output directory"),
    target: str | None = typer.Option(None, "--target", "-t", help="Target column name"),
    no_review: bool = typer.Option(False, "--no-review", help="Disable peer review"),
    no_literature: bool = typer.Option(False, "--no-literature", help="Skip literature review"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose logging"),
) -> None:
    """Run the full ML agent team pipeline on a dataset."""
    setup_logging("DEBUG" if verbose else "INFO")

    # Build config
    if config:
        pipeline_config = load_config(config)
    else:
        pipeline_config = PipelineConfig()

    pipeline_config.output_dir = output
    pipeline_config.enable_peer_review = not no_review
    pipeline_config.enable_literature_review = not no_literature

    if target:
        pipeline_config.data.target_column = target

    # Validate data path
    if not data.exists():
        console.print(f"[red]Error:[/red] Data file not found: {data}")
        raise typer.Exit(1)

    console.print(f"\n[bold]ML Agent Team Pipeline[/bold]")
    console.print(f"Problem: {problem}")
    console.print(f"Data: {data}")
    console.print(f"Output: {output}\n")

    # Run pipeline
    from .agents.orchestrator import OrchestratorAgent

    orchestrator = OrchestratorAgent(pipeline_config)

    try:
        result = asyncio.run(orchestrator.run(problem, str(data)))

        # Print results
        console.print("\n[bold green]Pipeline Complete![/bold green]\n")

        # Metrics table
        if result.metrics:
            table = Table(title="Model Performance")
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="green")
            for k, v in result.metrics.items():
                table.add_row(k, f"{v:.4f}")
            console.print(table)

        console.print(f"\nBest Model: [bold]{result.best_model_name}[/bold]")
        console.print(f"Acceptable: {'Yes' if result.is_acceptable else 'No'}")

        if result.report_path:
            console.print(f"Report: [link={result.report_path}]{result.report_path}[/link]")

    except Exception as e:
        console.print(f"\n[red]Pipeline failed:[/red] {e}")
        raise typer.Exit(1)


@app.command()
def list_agents() -> None:
    """List all available agents and their descriptions."""
    from .agents import (
        DataIngestionAgent,
        DiagnosisAgent,
        EDAAgent,
        EvaluationAgent,
        FeatureEngineeringAgent,
        LiteratureReviewAgent,
        ModelBuildingAgent,
        ModelSelectionAgent,
        OptimizationAgent,
        PeerReviewAgent,
        ProblemAnalysisAgent,
        ReportingAgent,
        TrainingAgent,
    )

    agents = [
        ProblemAnalysisAgent,
        LiteratureReviewAgent,
        DataIngestionAgent,
        EDAAgent,
        FeatureEngineeringAgent,
        ModelSelectionAgent,
        ModelBuildingAgent,
        TrainingAgent,
        EvaluationAgent,
        DiagnosisAgent,
        OptimizationAgent,
        ReportingAgent,
        PeerReviewAgent,
    ]

    table = Table(title="ML Agent Team — Available Agents")
    table.add_column("#", style="dim")
    table.add_column("Agent", style="cyan")
    table.add_column("Stage", style="green")
    table.add_column("Description")

    # We need dummy instances to read properties
    from .core.config import AgentConfig
    from .core.message_bus import MessageBus
    from .core.workflow_state import WorkflowState

    bus = MessageBus()
    state = WorkflowState()
    cfg = AgentConfig()

    for i, agent_cls in enumerate(agents, 1):
        agent = agent_cls(name="temp", config=cfg, message_bus=bus, workflow_state=state)
        table.add_row(str(i), agent_cls.__name__, agent.stage.value, agent.description)

    console.print(table)


if __name__ == "__main__":
    app()
