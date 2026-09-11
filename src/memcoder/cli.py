"""Command-line interface for solving tasks, inspecting memory, and benchmarking."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from memcoder.agent.workflow import MemCoderAgent
from memcoder.benchmark.runner import BenchmarkRunner
from memcoder.config import Settings
from memcoder.domain import MemoryCandidate, MemoryType
from memcoder.execution.sandbox import LocalPythonSandbox
from memcoder.memory.embeddings import HashingEmbedder, OpenAICompatibleEmbedder
from memcoder.memory.manager import MemoryManager
from memcoder.memory.store import SQLiteMemoryStore
from memcoder.models.demo import DemoStructuredModel
from memcoder.models.openai_compatible import OpenAICompatibleModel

app = typer.Typer(help="Memory-augmented coding agent and ablation benchmark.")
memory_app = typer.Typer(help="Inspect and manage long-term memory.")
app.add_typer(memory_app, name="memory")
console = Console()

DEMO_TASK = "Implement shortest_path(graph, start, target) for a non-negative weighted graph."
DEMO_TESTS = """\
from solution import shortest_path

graph = {
    "A": [("B", 5), ("C", 1)],
    "B": [("D", 1)],
    "C": [("B", 1), ("D", 8)],
    "D": [],
    "E": [],
}
assert shortest_path(graph, "A", "D") == 3
assert shortest_path(graph, "A", "A") == 0
assert shortest_path(graph, "A", "E") == float("inf")
"""


def _memory(settings: Settings) -> MemoryManager:
    store = SQLiteMemoryStore(settings.db_path)
    if settings.embedding_provider == "api":
        if not settings.api_key:
            raise typer.BadParameter("MEMCODER_API_KEY is required for API embeddings")
        embedder = OpenAICompatibleEmbedder(
            api_key=settings.api_key,
            model=settings.embedding_model,
            base_url=settings.base_url,
            dimensions=settings.embedding_dimensions,
        )
    else:
        embedder = HashingEmbedder(settings.embedding_dimensions)
    return MemoryManager(store, embedder)


def _live_agent(settings: Settings) -> MemCoderAgent:
    if not settings.api_key:
        raise typer.BadParameter("Set MEMCODER_API_KEY before using a live model")
    model = OpenAICompatibleModel(
        api_key=settings.api_key,
        model=settings.model,
        base_url=settings.base_url,
        api_mode=settings.api_mode,
    )
    return MemCoderAgent(
        model=model,
        memory=_memory(settings),
        executor=LocalPythonSandbox(timeout=settings.execution_timeout),
        top_k=settings.top_k,
    )


def _print_result(result, *, as_json: bool) -> None:
    if as_json:
        console.print_json(json.dumps(result.model_dump(mode="json"), ensure_ascii=False))
        return
    color = "green" if result.status == "success" else "red"
    console.print(f"[{color}]Status: {result.status}[/{color}]")
    console.print(f"Attempts: {result.attempts}")
    console.print(
        f"Memories retrieved/stored: "
        f"{len(result.memories_used)}/{len(result.stored_memory_ids)}"
    )
    console.print(f"Tokens: {result.usage.total_tokens}")
    console.rule("Final code")
    console.print(result.code)
    if result.execution:
        console.rule("Execution")
        console.print(result.execution.feedback)


@app.command()
def solve(
    task: Annotated[str, typer.Option("--task", "-t", help="Coding task description")],
    tests: Annotated[Path, typer.Option("--tests", exists=True, dir_okay=False)],
    memory: Annotated[bool, typer.Option("--memory/--no-memory")] = True,
    max_attempts: Annotated[int | None, typer.Option(min=1, max=10)] = None,
    as_json: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    """Solve one Python task against executable acceptance tests."""
    settings = Settings()
    agent = _live_agent(settings)
    result = agent.solve(
        task=task,
        tests=tests.read_text(encoding="utf-8"),
        memory_enabled=memory,
        max_attempts=max_attempts or settings.max_attempts,
    )
    _print_result(result, as_json=as_json)
    raise typer.Exit(code=0 if result.status == "success" else 1)


@app.command()
def demo(
    repeat: Annotated[int, typer.Option(min=1, max=5, help="Repeat to demonstrate recall")] = 2,
    db_path: Annotated[Path, typer.Option(help="Demo memory database")] = Path(
        ".memcoder/demo.sqlite3"
    ),
) -> None:
    """Run a deterministic offline repair-and-recall demonstration."""
    settings = Settings(db_path=db_path)
    agent = MemCoderAgent(
        model=DemoStructuredModel(),
        memory=_memory(settings),
        executor=LocalPythonSandbox(timeout=settings.execution_timeout),
        top_k=settings.top_k,
    )
    for index in range(1, repeat + 1):
        result = agent.solve(
            task=DEMO_TASK,
            tests=DEMO_TESTS,
            memory_enabled=True,
            max_attempts=3,
        )
        console.rule(f"Demo run {index}")
        console.print(
            f"status={result.status}, attempts={result.attempts}, "
            f"retrieved={len(result.memories_used)}, stored={len(result.stored_memory_ids)}"
        )


@app.command()
def benchmark(
    dataset: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    output: Annotated[Path, typer.Option()] = Path("artifacts/benchmark.json"),
    max_attempts: Annotated[int | None, typer.Option(min=1, max=10)] = None,
    repetitions: Annotated[int, typer.Option(min=1, max=20)] = 1,
    db_path: Annotated[Path, typer.Option(help="Fresh benchmark memory database")] = Path(
        ".memcoder/benchmark.sqlite3"
    ),
) -> None:
    """Run no-memory and memory-enabled modes on the same ordered dataset."""
    settings = Settings(db_path=db_path)

    def build_benchmark_agent(mode: str, repetition: int) -> MemCoderAgent:
        run_db = db_path.with_name(
            f"{db_path.stem}-{mode}-r{repetition}{db_path.suffix or '.sqlite3'}"
        )
        if run_db.exists():
            raise typer.BadParameter(
                f"Benchmark database already exists: {run_db}. Choose a fresh --db-path."
            )
        return _live_agent(settings.model_copy(update={"db_path": run_db}))

    runner = BenchmarkRunner(
        build_benchmark_agent, max_attempts=max_attempts or settings.max_attempts
    )
    report = runner.run(dataset, repetitions=repetitions)
    runner.write_json(report, output)
    table = Table("Mode", "Final pass", "First pass", "Avg attempts", "Tokens")
    for metric in report.metrics:
        table.add_row(
            metric.mode,
            f"{metric.final_pass_rate:.1%}",
            f"{metric.first_attempt_pass_rate:.1%}",
            f"{metric.average_attempts:.2f}",
            str(metric.total_input_tokens + metric.total_output_tokens),
        )
    console.print(table)
    console.print(f"Raw results: {output}")


@memory_app.command("list")
def list_memories(limit: Annotated[int, typer.Option(min=1, max=500)] = 50) -> None:
    settings = Settings()
    records = _memory(settings).list(limit=limit)
    table = Table("ID", "Type", "Task", "Uses", "Updated")
    for item in records:
        table.add_row(
            item.id[:8],
            item.memory_type.value,
            item.task_summary[:60],
            str(item.success_count),
            item.updated_at.isoformat(timespec="seconds"),
        )
    console.print(table)


@memory_app.command("search")
def search_memories(
    query: Annotated[str, typer.Argument()],
    limit: Annotated[int, typer.Option(min=1, max=20)] = 5,
) -> None:
    hits = _memory(Settings()).retrieve(query, top_k=limit)
    for hit in hits:
        console.print(
            f"[bold]{hit.memory.id}[/bold] score={hit.score:.3f} "
            f"semantic={hit.semantic_score:.3f} lexical={hit.lexical_score:.3f}"
        )
        console.print(hit.memory.content)


@memory_app.command("add")
def add_memory(
    memory_type: Annotated[MemoryType, typer.Argument()],
    task_summary: Annotated[str, typer.Argument()],
    content: Annotated[str, typer.Argument()],
    tags: Annotated[str, typer.Option(help="Comma-separated tags")] = "",
    importance: Annotated[float, typer.Option(min=0, max=1)] = 0.5,
) -> None:
    record = _memory(Settings()).write(
        MemoryCandidate(
            memory_type=memory_type,
            task_summary=task_summary,
            content=content,
            tags=[tag.strip() for tag in tags.split(",") if tag.strip()],
            importance=importance,
        )
    )
    console.print(f"Stored memory {record.id}")


@memory_app.command("update")
def update_memory(
    memory_id: Annotated[str, typer.Argument()],
    content: Annotated[str, typer.Option()],
) -> None:
    try:
        record = _memory(Settings()).update(memory_id, content=content)
    except KeyError as exc:
        raise typer.BadParameter(f"Memory not found: {memory_id}") from exc
    console.print(f"Updated memory {record.id}")


@memory_app.command("forget")
def forget_memory(memory_id: Annotated[str, typer.Argument()]) -> None:
    """Explicitly delete one memory by its full UUID."""
    deleted = _memory(Settings()).forget(memory_id)
    if not deleted:
        raise typer.BadParameter(f"Memory not found: {memory_id}")
    console.print(f"Forgot memory {memory_id}")


if __name__ == "__main__":
    app()
