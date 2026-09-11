# Benchmark methodology

## Research question

Does long-term memory improve a coding agent's effectiveness on related tasks under a fixed model,
prompt set, execution budget, and ordered dataset?

## Conditions

The runner evaluates two conditions:

- `without_memory`: retrieval and writes are disabled.
- `with_memory`: the same ordered tasks can retrieve and consolidate earlier experience.

Run the control condition first. Start with an empty benchmark database. Do not mix interactive memories
with benchmark memories.

## Required controls

Record the following with every published result:

- model identifier and provider;
- exact repository commit;
- dataset file hash;
- number of repetitions;
- maximum attempts and execution timeout;
- embedding provider and dimensions;
- temperature or sampling controls when supported;
- date of execution.

For nondeterministic models, run at least three repetitions. Treat task order as an experimental variable:
memory can only help later tasks if an earlier task produced relevant experience.

## Metrics

- Final pass rate.
- First-attempt pass rate.
- Average attempts.
- Average wall-clock duration.
- Total input and output tokens reported by the provider.
- Memory hit rate.

The raw JSON includes every task run. Do not publish only the aggregate table, and never replace actual
measurements with illustrative values.

## Interpretation limitations

A higher memory hit rate is not automatically beneficial. Retrieval can surface irrelevant or stale
experience. Inspect trajectories and compare failed tasks before concluding that memory caused an
improvement. The local hashing embedder is a baseline; results with API embeddings must be reported as a
separate configuration.

