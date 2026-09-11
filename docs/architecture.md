# Architecture

## Goals

MemCoder makes the agent loop and memory lifecycle small enough to inspect during an interview or
experiment. The system emphasizes traceability and ablation over broad IDE features.

## Workflow state

`AgentState` carries the task, executable tests, retrieved memories, plan, generated code, execution
result, reflection, attempt budget, token usage, and trajectory. Each LangGraph node appends a compact
event rather than logging hidden model reasoning.

The graph has seven nodes:

1. `retrieve`: search long-term memory, or return an empty list in the control condition.
2. `plan`: produce a typed implementation plan.
3. `generate`: return a complete Python module.
4. `execute`: run the module against supplied assertions.
5. `evaluate`: stop on success or exhausted budget.
6. `reflect`: diagnose the failure and choose whether to retry.
7. `store`: consolidate the episode and, after a successful repair, a procedural memory.

All LLM outputs are validated with Pydantic. Invalid structured data fails explicitly rather than being
silently interpreted.

## Memory model

The SQLite store supports three durable memory types:

- **Episodic**: what happened in a concrete task run.
- **Semantic**: stable programming facts or domain knowledge.
- **Procedural**: a reusable repair or problem-solving method.

Working memory is represented by the current LangGraph state and is not written to the long-term store.

## Lifecycle

### Write

Candidates are normalized and fingerprinted. Exact duplicates increment a reuse counter. Near duplicates
of the same type are consolidated when vector similarity exceeds the configured threshold.

### Retrieve

The baseline score is intentionally visible:

```text
score = 0.65 * semantic_similarity
      + 0.20 * lexical_overlap
      + 0.07 * importance
      + 0.04 * recency
      + 0.04 * reliability
```

The local feature-hashing embedder is deterministic and dependency-free. It is an offline baseline, not
a claim of parity with a trained embedding model. An OpenAI-compatible embedding adapter is available.

### Update

Editable fields are allowlisted. Changes to textual fields trigger a new embedding. Reuse count and last
access time remain explicit.

### Forget

Deletion requires the full memory identifier. Automated decay is deliberately deferred so that benchmark
behavior remains easy to audit.

## Execution boundary

Each attempt runs in a new temporary directory. The child receives a minimal environment without API
keys, uses isolated Python mode, and is constrained by wall-clock time and Unix resource limits. These
controls reduce accidents but do not replace a real container or VM boundary.

## Model boundary

The model interface accepts a system instruction, prompt, and Pydantic schema. Two HTTP transports are
supported:

- OpenAI Responses API with strict JSON Schema and `store=false`.
- OpenAI-compatible Chat Completions with JSON Schema response format.

Prompts label retrieved memories as fallible historical hints. This prevents stored text from being treated
as privileged system instructions.

