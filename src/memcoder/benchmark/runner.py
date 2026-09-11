"""Ablation runner comparing the same agent with and without memory."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from pathlib import Path
from statistics import mean
from typing import Literal

from pydantic import BaseModel, Field

from memcoder.agent.workflow import MemCoderAgent
from memcoder.domain import SolveResult


class BenchmarkTask(BaseModel):
    id: str
    prompt: str
    tests: str
    language: str = "python"
    tags: list[str] = Field(default_factory=list)


class TaskRun(BaseModel):
    task_id: str
    repetition: int
    mode: Literal["without_memory", "with_memory"]
    success: bool
    first_attempt_success: bool
    attempts: int
    duration_seconds: float
    input_tokens: int
    output_tokens: int
    memories_used: int


class ModeMetrics(BaseModel):
    mode: Literal["without_memory", "with_memory"]
    task_count: int
    final_pass_rate: float
    first_attempt_pass_rate: float
    average_attempts: float
    average_duration_seconds: float
    total_input_tokens: int
    total_output_tokens: int
    memory_hit_rate: float


class BenchmarkReport(BaseModel):
    dataset: str
    dataset_sha256: str
    repetitions: int
    runs: list[TaskRun]
    metrics: list[ModeMetrics]


def load_tasks(path: str | Path) -> list[BenchmarkTask]:
    source = Path(path)
    tasks = []
    for line_number, line in enumerate(source.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            tasks.append(BenchmarkTask.model_validate_json(line))
        except Exception as exc:
            raise ValueError(f"Invalid dataset line {line_number}: {exc}") from exc
    if not tasks:
        raise ValueError("Benchmark dataset is empty")
    return tasks


class BenchmarkRunner:
    def __init__(
        self,
        agent_factory: Callable[[str, int], MemCoderAgent],
        *,
        max_attempts: int = 3,
    ) -> None:
        self.agent_factory = agent_factory
        self.max_attempts = max_attempts

    def run(
        self,
        dataset_path: str | Path,
        *,
        modes: tuple[Literal["without_memory", "with_memory"], ...] = (
            "without_memory",
            "with_memory",
        ),
        repetitions: int = 1,
    ) -> BenchmarkReport:
        if repetitions < 1:
            raise ValueError("repetitions must be at least 1")
        tasks = load_tasks(dataset_path)
        runs: list[TaskRun] = []
        for mode in modes:
            enabled = mode == "with_memory"
            for repetition in range(1, repetitions + 1):
                agent = self.agent_factory(mode, repetition)
                for task in tasks:
                    result = agent.solve(
                        task=task.prompt,
                        tests=task.tests,
                        language=task.language,
                        memory_enabled=enabled,
                        max_attempts=self.max_attempts,
                    )
                    runs.append(self._task_run(task.id, mode, repetition, result))
        metrics = [self._metrics(mode, [run for run in runs if run.mode == mode]) for mode in modes]
        dataset_bytes = Path(dataset_path).read_bytes()
        return BenchmarkReport(
            dataset=str(dataset_path),
            dataset_sha256=hashlib.sha256(dataset_bytes).hexdigest(),
            repetitions=repetitions,
            runs=runs,
            metrics=metrics,
        )

    @staticmethod
    def _task_run(task_id: str, mode: str, repetition: int, result: SolveResult) -> TaskRun:
        executions = [event for event in result.trajectory if event["node"] == "execute"]
        return TaskRun(
            task_id=task_id,
            repetition=repetition,
            mode=mode,
            success=result.status == "success",
            first_attempt_success=bool(executions and executions[0].get("success")),
            attempts=result.attempts,
            duration_seconds=result.duration_seconds,
            input_tokens=result.usage.input_tokens,
            output_tokens=result.usage.output_tokens,
            memories_used=len(result.memories_used),
        )

    @staticmethod
    def _metrics(mode: str, runs: list[TaskRun]) -> ModeMetrics:
        count = len(runs)
        return ModeMetrics(
            mode=mode,
            task_count=count,
            final_pass_rate=sum(run.success for run in runs) / count,
            first_attempt_pass_rate=sum(run.first_attempt_success for run in runs) / count,
            average_attempts=mean(run.attempts for run in runs),
            average_duration_seconds=mean(run.duration_seconds for run in runs),
            total_input_tokens=sum(run.input_tokens for run in runs),
            total_output_tokens=sum(run.output_tokens for run in runs),
            memory_hit_rate=sum(run.memories_used > 0 for run in runs) / count,
        )

    @staticmethod
    def write_json(report: BenchmarkReport, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(report.model_dump(mode="json"), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
