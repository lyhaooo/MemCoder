from pathlib import Path

from memcoder.domain import MemoryCandidate, MemoryType
from memcoder.memory.embeddings import HashingEmbedder
from memcoder.memory.manager import MemoryManager
from memcoder.memory.store import SQLiteMemoryStore


def build_manager(path: Path) -> MemoryManager:
    return MemoryManager(SQLiteMemoryStore(path), HashingEmbedder(128))


def test_write_retrieve_update_forget_lifecycle(tmp_path: Path) -> None:
    manager = build_manager(tmp_path / "memory.sqlite3")
    candidate = MemoryCandidate(
        memory_type=MemoryType.PROCEDURAL,
        task_summary="Repair a Dijkstra implementation",
        content="heapq entries must be distance-node tuples",
        error_type="heap_tuple_order",
        root_cause="tuple fields were reversed",
        solution="push (distance, node)",
        tags=["python", "graph"],
        importance=0.9,
    )

    created = manager.write(candidate)
    duplicate = manager.write(candidate)

    assert created.id == duplicate.id
    assert duplicate.success_count == 2
    hits = manager.retrieve("Python shortest path heap ordering", top_k=3)
    assert hits
    assert hits[0].memory.id == created.id
    assert hits[0].score > 0

    updated = manager.update(created.id, solution="always use (distance, node)")
    assert updated.solution == "always use (distance, node)"
    assert manager.forget(created.id)
    assert manager.list() == []


def test_memory_type_filter(tmp_path: Path) -> None:
    manager = build_manager(tmp_path / "memory.sqlite3")
    manager.write(
        MemoryCandidate(
            memory_type=MemoryType.EPISODIC,
            task_summary="past run",
            content="a task passed",
        )
    )
    manager.write(
        MemoryCandidate(
            memory_type=MemoryType.SEMANTIC,
            task_summary="heap knowledge",
            content="heapq is a min heap",
        )
    )

    hits = manager.retrieve("heapq", memory_type=MemoryType.SEMANTIC)
    assert len(hits) == 1
    assert hits[0].memory.memory_type is MemoryType.SEMANTIC


def test_content_update_refreshes_deduplication_fingerprint(tmp_path: Path) -> None:
    manager = build_manager(tmp_path / "memory.sqlite3")
    original = MemoryCandidate(
        memory_type=MemoryType.SEMANTIC,
        task_summary="Priority queue behavior",
        content="heapq orders tuples by the first element",
    )
    created = manager.write(original)

    manager.update(
        created.id,
        content="A completely different verified observation about immutable mappings.",
    )
    restored = manager.write(original)

    assert restored.id != created.id
