"""Inspectable SQLite persistence for long-term memories."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from memcoder.domain import MemoryCandidate, MemoryRecord, MemoryType


def _utc_now() -> datetime:
    return datetime.now(UTC)


def memory_fingerprint(candidate: MemoryCandidate) -> str:
    payload = "|".join(
        [
            candidate.memory_type.value,
            candidate.task_summary.strip().lower(),
            candidate.content.strip().lower(),
            (candidate.error_type or "").strip().lower(),
        ]
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class SQLiteMemoryStore:
    """A small repository layer; every memory remains auditable as a row."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).expanduser()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA foreign_keys=ON")
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS memories (
                    id TEXT PRIMARY KEY,
                    memory_type TEXT NOT NULL,
                    task_summary TEXT NOT NULL,
                    content TEXT NOT NULL,
                    error_type TEXT,
                    root_cause TEXT,
                    solution TEXT,
                    code_pattern TEXT,
                    tags_json TEXT NOT NULL,
                    importance REAL NOT NULL,
                    embedding_json TEXT NOT NULL,
                    fingerprint TEXT NOT NULL UNIQUE,
                    success_count INTEGER NOT NULL DEFAULT 1,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    last_accessed_at TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_memories_type ON memories(memory_type);
                CREATE INDEX IF NOT EXISTS idx_memories_error ON memories(error_type);
                CREATE INDEX IF NOT EXISTS idx_memories_updated ON memories(updated_at);
                """
            )

    def insert(self, candidate: MemoryCandidate, embedding: list[float]) -> MemoryRecord:
        now = _utc_now()
        record = MemoryRecord(
            id=str(uuid.uuid4()),
            **candidate.model_dump(),
            embedding=embedding,
            created_at=now,
            updated_at=now,
        )
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO memories (
                    id, memory_type, task_summary, content, error_type, root_cause,
                    solution, code_pattern, tags_json, importance, embedding_json,
                    fingerprint, success_count, created_at, updated_at, last_accessed_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.id,
                    record.memory_type.value,
                    record.task_summary,
                    record.content,
                    record.error_type,
                    record.root_cause,
                    record.solution,
                    record.code_pattern,
                    json.dumps(record.tags, ensure_ascii=False),
                    record.importance,
                    json.dumps(record.embedding),
                    memory_fingerprint(candidate),
                    record.success_count,
                    record.created_at.isoformat(),
                    record.updated_at.isoformat(),
                    None,
                ),
            )
        return record

    def get(self, memory_id: str) -> MemoryRecord | None:
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM memories WHERE id = ?", (memory_id,)).fetchone()
        return self._row_to_record(row) if row else None

    def find_by_fingerprint(self, fingerprint: str) -> MemoryRecord | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM memories WHERE fingerprint = ?", (fingerprint,)
            ).fetchone()
        return self._row_to_record(row) if row else None

    def list(
        self, *, memory_type: MemoryType | None = None, limit: int = 100
    ) -> list[MemoryRecord]:
        query = "SELECT * FROM memories"
        params: list[Any] = []
        if memory_type:
            query += " WHERE memory_type = ?"
            params.append(memory_type.value)
        query += " ORDER BY updated_at DESC LIMIT ?"
        params.append(limit)
        with self._connect() as connection:
            rows = connection.execute(query, params).fetchall()
        return [self._row_to_record(row) for row in rows]

    def update(self, memory_id: str, **fields: Any) -> MemoryRecord:
        allowed = {
            "task_summary",
            "content",
            "error_type",
            "root_cause",
            "solution",
            "code_pattern",
            "tags",
            "importance",
            "embedding",
            "fingerprint",
            "success_count",
            "last_accessed_at",
        }
        unknown = set(fields) - allowed
        if unknown:
            raise ValueError(f"Unsupported update fields: {sorted(unknown)}")
        if not fields:
            record = self.get(memory_id)
            if record is None:
                raise KeyError(memory_id)
            return record

        encoded: dict[str, Any] = {}
        for key, value in fields.items():
            column = {"tags": "tags_json", "embedding": "embedding_json"}.get(key, key)
            if key in {"tags", "embedding"}:
                value = json.dumps(value, ensure_ascii=False)
            if isinstance(value, datetime):
                value = value.isoformat()
            encoded[column] = value
        if set(fields) != {"last_accessed_at"}:
            encoded["updated_at"] = _utc_now().isoformat()
        assignments = ", ".join(f"{key} = ?" for key in encoded)
        with self._connect() as connection:
            cursor = connection.execute(
                f"UPDATE memories SET {assignments} WHERE id = ?",  # noqa: S608
                [*encoded.values(), memory_id],
            )
            if cursor.rowcount == 0:
                raise KeyError(memory_id)
        record = self.get(memory_id)
        if record is None:
            raise KeyError(memory_id)
        return record

    def delete(self, memory_id: str) -> bool:
        with self._connect() as connection:
            cursor = connection.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
        return cursor.rowcount > 0

    def count(self) -> int:
        with self._connect() as connection:
            return int(connection.execute("SELECT COUNT(*) FROM memories").fetchone()[0])

    @staticmethod
    def _row_to_record(row: sqlite3.Row) -> MemoryRecord:
        return MemoryRecord(
            id=row["id"],
            memory_type=MemoryType(row["memory_type"]),
            task_summary=row["task_summary"],
            content=row["content"],
            error_type=row["error_type"],
            root_cause=row["root_cause"],
            solution=row["solution"],
            code_pattern=row["code_pattern"],
            tags=json.loads(row["tags_json"]),
            importance=row["importance"],
            embedding=json.loads(row["embedding_json"]),
            success_count=row["success_count"],
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
            last_accessed_at=(
                datetime.fromisoformat(row["last_accessed_at"])
                if row["last_accessed_at"]
                else None
            ),
        )
