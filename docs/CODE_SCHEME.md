# MACS Codebase Scheme

This document is a plain-language map of how MACS runs.

## High-Level Flow

```
[User]
  |
  v
[CLI / GUI / Adaptive Runner]
  |
  v
[GraphOrchestrator]
  |  - loads config.json (via graph_loader)
  |  - validates graph (via graph_validator)
  |  - builds DAG (via graph_builder)
  |  - executes agents (via execution_engine)
  |  - emits events
  |
  +---> [EventBus] -> [EventLogger, MetricsCollector]
  |
  v
[Agents]
  |
  v
[Output Files + Metrics + Visualization]
```

## Module Overview

The orchestration logic is split into focused modules:

| Module | Purpose |
|--------|---------|
| `graph_loader.py` | Load and normalize config files with environment variable substitution |
| `graph_validator.py` | Schema validation, cycle detection, and comprehensive graph checks |
| `graph_builder.py` | Build adjacency structures and compute topological order |
| `execution_engine.py` | Node execution with retry, loop, and parallel support |
| `context_injection.py` | Prepare inputs for agents based on capabilities |
| `artifact_writer.py` | Pluggable output writers for summaries, briefs, etc. |
| `contracts.py` | AgentCapabilities dataclass and input/output contracts |
| `events.py` | Event bus and event type definitions |
| `event_subscribers.py` | EventLogger and MetricsCollector handlers |

## Graph Validation Pipeline

Before execution, the graph is validated through these steps:

1. **Schema Validation** - Pydantic models check types and required fields
2. **Reference Validation** - All edge endpoints must exist as nodes
3. **Cycle Detection** - DFS ensures the graph is a DAG
4. **Agent Type Validation** - Node types must be registered agents
5. **Data Mapping Validation** - Check consistency of input/output keys

Use `validate_graph_comprehensive()` for full validation with structured results:

```python
from graph_validator import validate_graph_comprehensive

result = validate_graph_comprehensive(graph_def, check_agent_types=True)
if result.valid:
    orchestrator.run(result.validated_graph)
else:
    for error in result.errors:
        print(f"ERROR: {error}")
```

## Execution Steps

1. You start MACS from the GUI (`gui.py`), the CLI (`cli.py`), or the adaptive
   runner (`adaptive/adaptive_graph_runner.py`).
2. The entrypoint reads `config.json` and validates the graph definition.
3. The orchestrator builds a DAG from `graph_definition` and computes execution
   order using topological sort.
4. Each node maps to a Python class in `agents/` (or a plugin) and an instance
   is created with the LLM client.
5. The orchestrator executes nodes in order. Each agent receives inputs based
   on edge `data_mapping` and agent capabilities.
6. While nodes run, events are emitted to subscribers (logging, metrics).
7. Each agent returns a dictionary. Keys are mapped to downstream node inputs.
8. Output files are written via `artifact_writer.py` into the project directory.

## Graph Visualization

MACS generates interactive HTML graphs using PyVis:

| File | Description |
|------|-------------|
| `graph_initial.html` | Graph before execution starts |
| `graph_{node}.html` | Graph during execution (current node highlighted) |
| `graph_final.html` | Graph after workflow completes |

**Features:**
- Hierarchical top-down layout showing DAG flow
- Click any node to view agent details in an info panel
- Color-coded execution states (pending, executing, completed, failed)
- Hover tooltips with agent type and metrics

## Where to Find the Code

- `multi_agent_llm_system.py` - GraphOrchestrator and main entrypoint
- `agents/` - Built-in agent classes
- `agents/registry.py` - Agent registration and capabilities
- `agent_plugins/` - Optional plugin agents
- `graph_*.py` - Graph building and validation modules
- `execution_engine.py` - Node execution with loop/retry support
- `context_injection.py` - Input preparation based on capabilities
- `artifact_writer.py` - Output file writers
- `events.py` and `event_subscribers.py` - Event system
- `adaptive/` - Adaptive cycle for graph evolution
- `config_schema.py` - Pydantic schema for config validation

## Agent Capabilities System

Agents declare requirements via class-level attributes in `base_agent.py`:

```python
class MyAgent(Agent):
    REQUIRES_PROJECT_DIR = True      # Needs project_base_output_dir
    CONSUMES_ALL_OUTPUTS = False     # Needs all_agent_outputs
    REQUIRED_INPUTS = ["input_key"]  # Must be provided
    OPTIONAL_INPUTS = ["optional_key"]  # May be provided
```

The orchestrator uses these via `agent.get_capabilities()` to:
- Validate that required inputs are mapped
- Inject project directory when needed
- Aggregate all outputs for integrator agents

## Adaptive Evolution Loop

The adaptive runner repeats: run → evaluate → mutate

1. Run the current graph configuration
2. Score results with evaluation plugins
3. If score is below threshold, mutate the graph config
4. Save per-cycle metrics in `adaptive_metrics/`

This lets MACS improve workflows without changing agent code.
