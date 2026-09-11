from pathlib import Path

from memcoder.agent.workflow import MemCoderAgent
from memcoder.benchmark.runner import BenchmarkRunner
from memcoder.cli import DEMO_TASK, DEMO_TESTS
from memcoder.execution.sandbox import LocalPythonSandbox
from memcoder.memory.embeddings import HashingEmbedder
from memcoder.memory.manager import MemoryManager
from memcoder.memory.store import SQLiteMemoryStore
from memcoder.models.demo import DemoStructuredModel


def test_benchmark_reports_ablation_metrics(tmp_path: Path) -> None:
    dataset = tmp_path / "tasks.jsonl"
    rows = [
        {"id": "dijkstra-1", "prompt": DEMO_TASK, "tests": DEMO_TESTS},
        {"id": "dijkstra-2", "prompt": DEMO_TASK, "tests": DEMO_TESTS},
    ]
    dataset.write_text("\n".join(__import__("json").dumps(row) for row in rows), encoding="utf-8")
    def agent_factory(mode: str, repetition: int) -> MemCoderAgent:
        memory = MemoryManager(
            SQLiteMemoryStore(tmp_path / f"{mode}-{repetition}.sqlite3"),
            HashingEmbedder(128),
        )
        return MemCoderAgent(
            model=DemoStructuredModel(),
            memory=memory,
            executor=LocalPythonSandbox(timeout=2),
        )

    report = BenchmarkRunner(agent_factory).run(dataset)

    without, with_memory = report.metrics
    assert without.mode == "without_memory"
    assert without.first_attempt_pass_rate == 0
    assert with_memory.mode == "with_memory"
    assert with_memory.first_attempt_pass_rate == 0.5
    assert with_memory.average_attempts < without.average_attempts
    assert report.dataset_sha256
    assert report.repetitions == 1
