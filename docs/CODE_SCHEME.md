# MACS Codebase Scheme

This document is a plain-language map of how MACS runs.

## High-Level Flow (Simple)

```
[User]
  |
  v
[CLI / GUI / Adaptive Runner]
  |
  v
[GraphOrchestrator]
  |  - loads config.json
  |  - builds DAG
  |  - executes agents in order
  |  - emits events
  |
  +--> [EventBus] -> [EventLogger, MetricsCollector]
  |
  v
[Agents]
  |
  v
[Output Files + Metrics]
```

## Execution Steps (Beginner Version)

1. You start MACS from the GUI (`gui.py`), the CLI (`cli.py`), or the adaptive runner
   (`adaptive/adaptive_graph_runner.py`).
2. The entrypoint reads `config.json` (or the file you pass in) and sends the
   settings to the orchestrator.
3. The orchestrator builds a directed acyclic graph (DAG) of agent nodes from the
   `graph_definition` section in the config.
4. Each node is mapped to a Python class in `agents/` (or a plugin) and an
   instance is created.
5. The orchestrator executes nodes in a safe order (topological order) so that
   a node only runs after its required inputs are ready.
6. While nodes run, the orchestrator emits events to the event bus. Subscribers
   such as `EventLogger` (logs) and `MetricsCollector` (writes `node_metrics.json`)
   listen to those events.
7. Each agent returns a dictionary. The orchestrator maps the keys into inputs
   for downstream nodes based on the edge `data_mapping` in the config.
8. The orchestrator writes output files (summaries, briefs, hypotheses, etc.)
   into your project output directory.

## Where to Find the Code

- `multi_agent_llm_system.py` - orchestration engine (GraphOrchestrator) and
  `run_project_orchestration` entrypoint.
- `agents/` - built-in agent classes (PDF loader, summarizer, memory, etc.).
- `agent_plugins/` - optional plugin agents.
- `events.py` and `event_subscribers.py` - event bus and subscriber handlers.
- `adaptive/` - adaptive cycle that evaluates and mutates the graph.
- `utils.py` and `config_schema.py` - config loading, validation, and helpers.

## Adaptive Evolution Loop

The adaptive runner repeats a cycle of run -> evaluate -> mutate:

1. Run the current graph.
2. Score results with one or more evaluation plugins.
3. If the score is below the threshold, mutate the graph config and try again.
4. Save per-cycle metrics in `adaptive_metrics/` under the output directory.

This lets MACS improve a workflow without changing agent code.
