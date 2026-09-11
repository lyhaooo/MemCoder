"""Auditable LangGraph workflow for the memory-augmented coding agent."""

from __future__ import annotations

import time
from typing import Any, Literal

from langgraph.graph import END, START, StateGraph

from memcoder.agent.prompts import (
    CODER_SYSTEM,
    PLANNER_SYSTEM,
    REFLECTION_SYSTEM,
    code_prompt,
    plan_prompt,
    reflection_prompt,
)
from memcoder.domain import (
    AgentState,
    CodeOutput,
    MemoryCandidate,
    MemoryType,
    PlanOutput,
    ReflectionOutput,
    SolveResult,
    TokenUsage,
)
from memcoder.execution.sandbox import LocalPythonSandbox
from memcoder.memory.manager import MemoryManager
from memcoder.models.base import StructuredModel


class MemCoderAgent:
    """Generate, execute, reflect, retry, and retain useful experience."""

    def __init__(
        self,
        *,
        model: StructuredModel,
        memory: MemoryManager,
        executor: LocalPythonSandbox,
        top_k: int = 5,
    ) -> None:
        self.model = model
        self.memory = memory
        self.executor = executor
        self.top_k = top_k
        self.graph = self._build_graph()

    def _build_graph(self):
        graph = StateGraph(AgentState)
        graph.add_node("retrieve", self._retrieve)
        graph.add_node("plan", self._plan)
        graph.add_node("generate", self._generate)
        graph.add_node("execute", self._execute)
        graph.add_node("evaluate", self._evaluate)
        graph.add_node("reflect", self._reflect)
        graph.add_node("store", self._store)
        graph.add_edge(START, "retrieve")
        graph.add_edge("retrieve", "plan")
        graph.add_edge("plan", "generate")
        graph.add_edge("generate", "execute")
        graph.add_edge("execute", "evaluate")
        graph.add_conditional_edges(
            "evaluate",
            self._route_after_evaluation,
            {"reflect": "reflect", "store": "store"},
        )
        graph.add_conditional_edges(
            "reflect",
            self._route_after_reflection,
            {"generate": "generate", "store": "store"},
        )
        graph.add_edge("store", END)
        return graph.compile()

    def solve(
        self,
        *,
        task: str,
        tests: str,
        memory_enabled: bool = True,
        max_attempts: int = 3,
        language: str = "python",
    ) -> SolveResult:
        started = time.perf_counter()
        initial: AgentState = {
            "task": task,
            "tests": tests,
            "language": language,
            "memory_enabled": memory_enabled,
            "max_attempts": max_attempts,
            "attempt": 0,
            "memories": [],
            "code": "",
            "status": "running",
            "usage": TokenUsage(),
            "stored_memory_ids": [],
            "trajectory": [],
        }
        state = self.graph.invoke(initial)
        duration = time.perf_counter() - started
        execution = state.get("execution")
        return SolveResult(
            task=task,
            status=state["status"],
            code=state.get("code", ""),
            attempts=state.get("attempt", 0),
            memories_used=[hit.memory.id for hit in state.get("memories", [])],
            stored_memory_ids=state.get("stored_memory_ids", []),
            execution=execution,
            usage=state.get("usage", TokenUsage()),
            duration_seconds=duration,
            trajectory=state.get("trajectory", []),
        )

    def _retrieve(self, state: AgentState) -> dict[str, Any]:
        hits = (
            self.memory.retrieve(state["task"], top_k=self.top_k)
            if state["memory_enabled"]
            else []
        )
        return {
            "memories": hits,
            "trajectory": self._event(state, "retrieve", count=len(hits)),
        }

    def _plan(self, state: AgentState) -> dict[str, Any]:
        plan, result = self.model.generate(
            system=PLANNER_SYSTEM,
            prompt=plan_prompt(state["task"], state["tests"], state["memories"]),
            schema=PlanOutput,
        )
        return {
            "plan": plan,
            "usage": state["usage"] + result.usage,
            "trajectory": self._event(state, "plan", summary=plan.summary),
        }

    def _generate(self, state: AgentState) -> dict[str, Any]:
        output, result = self.model.generate(
            system=CODER_SYSTEM,
            prompt=code_prompt(
                state["task"],
                state["tests"],
                state["plan"],
                state["memories"],
                state.get("execution"),
                state.get("reflection"),
            ),
            schema=CodeOutput,
        )
        attempt = state["attempt"] + 1
        return {
            "code": output.code,
            "attempt": attempt,
            "usage": state["usage"] + result.usage,
            "trajectory": self._event(state, "generate", attempt=attempt),
        }

    def _execute(self, state: AgentState) -> dict[str, Any]:
        result = self.executor.execute(state["code"], state["tests"])
        return {
            "execution": result,
            "trajectory": self._event(
                state,
                "execute",
                success=result.success,
                return_code=result.return_code,
                duration_seconds=round(result.duration_seconds, 4),
            ),
        }

    def _evaluate(self, state: AgentState) -> dict[str, Any]:
        if state["execution"].success:
            status: Literal["running", "success", "failed"] = "success"
        elif state["attempt"] >= state["max_attempts"]:
            status = "failed"
        else:
            status = "running"
        return {
            "status": status,
            "trajectory": self._event(state, "evaluate", status=status),
        }

    @staticmethod
    def _route_after_evaluation(state: AgentState) -> Literal["reflect", "store"]:
        return "store" if state["status"] in {"success", "failed"} else "reflect"

    @staticmethod
    def _route_after_reflection(state: AgentState) -> Literal["generate", "store"]:
        reflection = state["reflection"]
        return "generate" if reflection.should_retry else "store"

    def _reflect(self, state: AgentState) -> dict[str, Any]:
        reflection, result = self.model.generate(
            system=REFLECTION_SYSTEM,
            prompt=reflection_prompt(state["task"], state["code"], state["execution"]),
            schema=ReflectionOutput,
        )
        max_attempts = state["max_attempts"] if reflection.should_retry else state["attempt"]
        status = state["status"] if reflection.should_retry else "failed"
        return {
            "reflection": reflection,
            "max_attempts": max_attempts,
            "status": status,
            "usage": state["usage"] + result.usage,
            "trajectory": self._event(
                state,
                "reflect",
                error_type=reflection.error_type,
                should_retry=reflection.should_retry,
            ),
        }

    def _store(self, state: AgentState) -> dict[str, Any]:
        if not state["memory_enabled"]:
            return {"trajectory": self._event(state, "store", skipped=True)}

        execution = state.get("execution")
        reflection = state.get("reflection")
        outcome = state["status"]
        content = (
            f"Task ended with {outcome} after {state['attempt']} attempt(s). "
            f"Final execution: {execution.feedback if execution else 'not available'}"
        )
        candidate = MemoryCandidate(
            memory_type=MemoryType.EPISODIC,
            task_summary=state["task"],
            content=content,
            error_type=reflection.error_type if reflection else None,
            root_cause=reflection.diagnosis if reflection else None,
            solution=(
                reflection.fix_strategy
                if reflection
                else "Generated code passed the supplied tests."
            ),
            code_pattern=state.get("code", "")[:1500],
            tags=[state.get("language", "python"), outcome],
            importance=0.8 if outcome == "success" else 0.35,
        )
        stored = self.memory.write(candidate)
        stored_ids = [stored.id]
        if reflection and outcome == "success":
            procedure = self.memory.write(
                MemoryCandidate(
                    memory_type=MemoryType.PROCEDURAL,
                    task_summary=f"How to recover from {reflection.error_type} in Python code",
                    content=reflection.fix_strategy,
                    error_type=reflection.error_type,
                    root_cause=reflection.diagnosis,
                    solution=reflection.fix_strategy,
                    tags=[state.get("language", "python"), "repair", reflection.error_type],
                    importance=0.7,
                )
            )
            stored_ids.append(procedure.id)
        return {
            "stored_memory_ids": stored_ids,
            "trajectory": self._event(state, "store", memory_ids=stored_ids),
        }

    @staticmethod
    def _event(state: AgentState, node: str, **details: Any) -> list[dict[str, Any]]:
        return [*state.get("trajectory", []), {"node": node, **details}]
