"""Typed domain objects shared by the agent, memory, and benchmark layers."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Literal, TypedDict

from pydantic import BaseModel, Field


class MemoryType(StrEnum):
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    PROCEDURAL = "procedural"


class TokenUsage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def __add__(self, other: object) -> TokenUsage:
        if not isinstance(other, TokenUsage):
            return NotImplemented
        return TokenUsage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
        )


class ModelResult(BaseModel):
    data: dict[str, Any]
    usage: TokenUsage = Field(default_factory=TokenUsage)
    raw_text: str = ""


class PlanOutput(BaseModel):
    summary: str
    steps: list[str] = Field(min_length=1, max_length=10)
    target_language: str = "python"


class CodeOutput(BaseModel):
    code: str
    explanation: str = ""


class ReflectionOutput(BaseModel):
    diagnosis: str
    fix_strategy: str
    error_type: str = "unknown"
    should_retry: bool = True


class MemoryCandidate(BaseModel):
    memory_type: MemoryType
    task_summary: str
    content: str
    error_type: str | None = None
    root_cause: str | None = None
    solution: str | None = None
    code_pattern: str | None = None
    tags: list[str] = Field(default_factory=list)
    importance: float = Field(default=0.5, ge=0, le=1)


class MemoryRecord(MemoryCandidate):
    id: str
    embedding: list[float]
    success_count: int = 1
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    last_accessed_at: datetime | None = None


class MemoryHit(BaseModel):
    memory: MemoryRecord
    score: float
    semantic_score: float
    lexical_score: float


class ExecutionResult(BaseModel):
    success: bool
    return_code: int
    stdout: str = ""
    stderr: str = ""
    duration_seconds: float = 0.0
    timed_out: bool = False

    @property
    def feedback(self) -> str:
        status = "passed" if self.success else "failed"
        details = self.stderr.strip() or self.stdout.strip() or "No output"
        return f"Execution {status} (return_code={self.return_code}):\n{details}"


class SolveResult(BaseModel):
    task: str
    status: Literal["success", "failed"]
    code: str
    attempts: int
    memories_used: list[str]
    stored_memory_ids: list[str]
    execution: ExecutionResult | None
    usage: TokenUsage
    duration_seconds: float
    trajectory: list[dict[str, Any]]


class AgentState(TypedDict, total=False):
    task: str
    tests: str
    language: str
    memory_enabled: bool
    max_attempts: int
    attempt: int
    memories: list[MemoryHit]
    plan: PlanOutput
    code: str
    execution: ExecutionResult
    reflection: ReflectionOutput
    status: Literal["running", "success", "failed"]
    usage: TokenUsage
    stored_memory_ids: list[str]
    trajectory: list[dict[str, Any]]
