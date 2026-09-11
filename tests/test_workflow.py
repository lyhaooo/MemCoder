from pathlib import Path

from memcoder.agent.workflow import MemCoderAgent
from memcoder.cli import DEMO_TASK, DEMO_TESTS
from memcoder.execution.sandbox import LocalPythonSandbox
from memcoder.memory.embeddings import HashingEmbedder
from memcoder.memory.manager import MemoryManager
from memcoder.memory.store import SQLiteMemoryStore
from memcoder.models.demo import DemoStructuredModel


def build_agent(tmp_path: Path) -> MemCoderAgent:
    manager = MemoryManager(
        SQLiteMemoryStore(tmp_path / "memory.sqlite3"), HashingEmbedder(128)
    )
    return MemCoderAgent(
        model=DemoStructuredModel(),
        memory=manager,
        executor=LocalPythonSandbox(timeout=2),
        top_k=5,
    )


def test_cold_repair_then_warm_recall(tmp_path: Path) -> None:
    agent = build_agent(tmp_path)

    cold = agent.solve(task=DEMO_TASK, tests=DEMO_TESTS, memory_enabled=True)
    warm = agent.solve(task=DEMO_TASK, tests=DEMO_TESTS, memory_enabled=True)

    assert cold.status == "success"
    assert cold.attempts == 2
    assert len(cold.stored_memory_ids) == 2
    assert any(event["node"] == "reflect" for event in cold.trajectory)
    assert warm.status == "success"
    assert warm.attempts == 1
    assert warm.memories_used


def test_no_memory_mode_neither_reads_nor_writes(tmp_path: Path) -> None:
    agent = build_agent(tmp_path)
    result = agent.solve(task=DEMO_TASK, tests=DEMO_TESTS, memory_enabled=False)

    assert result.status == "success"
    assert result.memories_used == []
    assert result.stored_memory_ids == []

