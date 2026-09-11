"""Deterministic offline model used only for the product demo."""

from __future__ import annotations

from typing import TypeVar

from pydantic import BaseModel

from memcoder.domain import (
    CodeOutput,
    ModelResult,
    PlanOutput,
    ReflectionOutput,
)

SchemaT = TypeVar("SchemaT", bound=BaseModel)

_BUGGY_DIJKSTRA = '''
import heapq

def shortest_path(graph, start, target):
    distances = {node: float("inf") for node in graph}
    distances[start] = 0
    queue = [(0, start)]
    while queue:
        distance, node = heapq.heappop(queue)
        if node == target:
            return distance
        for neighbor, weight in graph[node]:
            candidate = distance + weight
            if candidate < distances[neighbor]:
                distances[neighbor] = candidate
                # Deliberate demo defect: tuple order is reversed.
                heapq.heappush(queue, (neighbor, candidate))
    return float("inf")
'''.strip()

_CORRECT_DIJKSTRA = '''
import heapq

def shortest_path(graph, start, target):
    if start not in graph or target not in graph:
        return float("inf")
    distances = {node: float("inf") for node in graph}
    distances[start] = 0
    queue = [(0, start)]
    while queue:
        distance, node = heapq.heappop(queue)
        if distance != distances[node]:
            continue
        if node == target:
            return distance
        for neighbor, weight in graph[node]:
            candidate = distance + weight
            if candidate < distances[neighbor]:
                distances[neighbor] = candidate
                heapq.heappush(queue, (candidate, neighbor))
    return float("inf")
'''.strip()


class DemoStructuredModel:
    """Makes the first cold attempt fail and lets memory prevent the repeat."""

    def generate(
        self,
        *,
        system: str,
        prompt: str,
        schema: type[SchemaT],
    ) -> tuple[SchemaT, ModelResult]:
        if schema is PlanOutput:
            value: BaseModel = PlanOutput(
                summary="Implement Dijkstra with a min-heap and explicit unreachable handling.",
                steps=[
                    "Initialize distances and a heap of (distance, node) tuples",
                    "Relax outgoing edges while skipping stale heap entries",
                    "Return infinity when the target is unreachable",
                ],
            )
        elif schema is ReflectionOutput:
            value = ReflectionOutput(
                diagnosis=(
                    "heapq entries were pushed as (node, distance), breaking priority ordering"
                ),
                fix_strategy=(
                    "Always push and pop (distance, node) tuples and ignore stale entries."
                ),
                error_type="heap_tuple_order",
                should_retry=True,
            )
        elif schema is CodeOutput:
            has_memory = "Memory 1 (" in prompt
            is_retry = "PREVIOUS EXECUTION" in prompt
            value = CodeOutput(
                code=_CORRECT_DIJKSTRA if has_memory or is_retry else _BUGGY_DIJKSTRA,
                explanation=(
                    "Use the recalled heap invariant." if has_memory else "Implement the plan."
                ),
            )
        else:
            raise TypeError(f"Unsupported demo schema: {schema.__name__}")
        typed = schema.model_validate(value.model_dump())
        result = ModelResult(data=typed.model_dump(mode="json"), raw_text=typed.model_dump_json())
        return typed, result
