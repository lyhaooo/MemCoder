"""Memory write, retrieve, update, and forget lifecycle."""

from __future__ import annotations

import math
import re
from datetime import UTC, datetime
from typing import Any

from memcoder.domain import MemoryCandidate, MemoryHit, MemoryRecord, MemoryType
from memcoder.memory.embeddings import Embedder, cosine_similarity
from memcoder.memory.store import SQLiteMemoryStore, memory_fingerprint


class MemoryManager:
    """Coordinates deduplicated writes and explainable hybrid retrieval."""

    def __init__(
        self,
        store: SQLiteMemoryStore,
        embedder: Embedder,
        *,
        deduplication_threshold: float = 0.96,
    ) -> None:
        self.store = store
        self.embedder = embedder
        self.deduplication_threshold = deduplication_threshold

    def write(self, candidate: MemoryCandidate) -> MemoryRecord:
        """Insert a memory or consolidate an exact/near duplicate."""
        fingerprint = memory_fingerprint(candidate)
        exact = self.store.find_by_fingerprint(fingerprint)
        if exact:
            return self.store.update(exact.id, success_count=exact.success_count + 1)

        embedding = self.embedder.embed(self._candidate_text(candidate))
        for existing in self.store.list(memory_type=candidate.memory_type, limit=500):
            similarity = cosine_similarity(embedding, existing.embedding)
            if similarity >= self.deduplication_threshold:
                return self.store.update(
                    existing.id,
                    content=candidate.content,
                    root_cause=candidate.root_cause or existing.root_cause,
                    solution=candidate.solution or existing.solution,
                    code_pattern=candidate.code_pattern or existing.code_pattern,
                    tags=sorted(set(existing.tags + candidate.tags)),
                    importance=max(existing.importance, candidate.importance),
                    embedding=embedding,
                    fingerprint=fingerprint,
                    success_count=existing.success_count + 1,
                )
        return self.store.insert(candidate, embedding)

    def retrieve(
        self,
        query: str,
        *,
        top_k: int = 5,
        memory_type: MemoryType | None = None,
        error_type: str | None = None,
        min_score: float = 0.05,
    ) -> list[MemoryHit]:
        """Combine vector similarity, lexical overlap, importance, and recency."""
        query_embedding = self.embedder.embed(query)
        query_terms = self._terms(query)
        now = datetime.now(UTC)
        hits: list[MemoryHit] = []
        for memory in self.store.list(memory_type=memory_type, limit=1000):
            if error_type and memory.error_type != error_type:
                continue
            semantic = max(0.0, cosine_similarity(query_embedding, memory.embedding))
            memory_terms = self._terms(self._record_text(memory))
            lexical = len(query_terms & memory_terms) / max(1, len(query_terms | memory_terms))
            age_days = max(0.0, (now - memory.updated_at).total_seconds() / 86400)
            recency = math.exp(-age_days / 90)
            reliability = min(1.0, math.log1p(memory.success_count) / math.log(6))
            score = (
                0.65 * semantic
                + 0.20 * lexical
                + 0.07 * memory.importance
                + 0.04 * recency
                + 0.04 * reliability
            )
            if score >= min_score:
                hits.append(
                    MemoryHit(
                        memory=memory,
                        score=score,
                        semantic_score=semantic,
                        lexical_score=lexical,
                    )
                )
        hits.sort(key=lambda hit: hit.score, reverse=True)
        selected = hits[:top_k]
        accessed_at = datetime.now(UTC)
        for hit in selected:
            self.store.update(hit.memory.id, last_accessed_at=accessed_at)
        return selected

    def update(self, memory_id: str, **fields: Any) -> MemoryRecord:
        """Update an existing memory and re-embed it when its text changes."""
        current = self.store.get(memory_id)
        if current is None:
            raise KeyError(memory_id)
        merged = current.model_dump(
            exclude={
                "id",
                "embedding",
                "created_at",
                "updated_at",
                "last_accessed_at",
                "success_count",
            }
        )
        merged.update(fields)
        candidate = MemoryCandidate.model_validate(merged)
        update_fields = dict(fields)
        if {"task_summary", "content", "root_cause", "solution", "code_pattern"} & fields.keys():
            update_fields["embedding"] = self.embedder.embed(self._candidate_text(candidate))
        if {"task_summary", "content", "error_type"} & fields.keys():
            update_fields["fingerprint"] = memory_fingerprint(candidate)
        return self.store.update(memory_id, **update_fields)

    def forget(self, memory_id: str) -> bool:
        """Delete a memory by explicit identifier."""
        return self.store.delete(memory_id)

    def list(self, *, limit: int = 100) -> list[MemoryRecord]:
        return self.store.list(limit=limit)

    @staticmethod
    def _candidate_text(candidate: MemoryCandidate) -> str:
        return "\n".join(
            filter(
                None,
                [
                    candidate.task_summary,
                    candidate.content,
                    candidate.error_type,
                    candidate.root_cause,
                    candidate.solution,
                    candidate.code_pattern,
                    " ".join(candidate.tags),
                ],
            )
        )

    @staticmethod
    def _record_text(memory: MemoryRecord) -> str:
        return MemoryManager._candidate_text(MemoryCandidate.model_validate(memory.model_dump()))

    @staticmethod
    def _terms(text: str) -> set[str]:
        return set(re.findall(r"[a-z0-9_]+|[\u4e00-\u9fff]", text.lower()))
