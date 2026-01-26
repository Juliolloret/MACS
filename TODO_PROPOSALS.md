# Proposed TODOs for sustainability and extensibility

- [ ] Decompose `multi_agent_llm_system.py` into smaller units (graph construction/validation, execution engine, visualization, artifact persistence). This removes the 800+ line single class, clarifies responsibilities, and lets tests target each piece independently.
- [ ] Replace ad-hoc dict I/O between nodes with typed contracts (e.g., dataclasses or Pydantic models for inputs/results with `data`, `error`, `metadata`). This reduces fragile key lookups (`*_error`, `*_source`) and makes adding new agent outputs safer.
- [ ] Introduce agent capability metadata so the orchestrator does not special-case types (e.g., ExperimentalDataLoaderAgent path injection, memory/writer project paths, KnowledgeIntegrator aggregations). Agents would declare required context (needs_project_dir, consumes_all_outputs, optional_inputs) and the orchestrator injects based on that contract.
- [ ] Add a central LLM/config factory (`llm_factory.py`) that resolves client type, API keys, timeouts, and caches. Reuse it from `multi_agent_llm_system.py`, `adaptive/adaptive_graph_runner.py`, and GUI entrypoints to avoid duplicated setup logic and inconsistent error handling.
- [ ] Extract artifact writing from `_save_consolidated_outputs` into pluggable writers keyed off graph config (mapping output keys to folders/templates). This avoids editing the orchestrator whenever a new agent needs to persist files and makes output destinations configurable.
- [ ] Standardize logging/metrics: move the callback + `log_status` pattern to a lightweight logger with structured context (run_id, node_id, attempt, tokens). Emit metrics events instead of writing `node_metrics.json` manually to simplify observability and downstream dashboards.
- [ ] Create shared LLM helper mixins for agents (temperature/reasoning/verbosity handling, truncation utilities, common prompt assembly) to remove duplicated blocks across summarizer, synthesizer, integrator, and deep research agents.
- [ ] Split `gui.py` into view/controller/service modules that reuse the same orchestration service as `cli.py`, so UI logic is separate from execution logic and new UI surfaces (web/CLI) can share code.
- [ ] Strengthen configuration schema: validate `data_mapping` keys against declared agent outputs, support optional/required inputs per node, and surface friendly errors before runtime. Include environment variable substitution for secrets to avoid placeholders in `config.json`.
- [ ] Add higher-level contract tests around plugin discovery and orchestrator capability negotiation to keep the extension points stable as the structure is refactored.
- [x] Refresh documentation for beginners: rewrite `README.md`, `AGENTS.md`, `docs/*`, `memory.md`, and `CONTRIBUTING.md` with step-by-step setup, config guidance, testing, and troubleshooting.
- [x] Migrate PDF ingestion from `PyPDF2` to `pypdf` to remove deprecation warnings and align with the maintained library. Update imports, adjust any API differences, and refresh tests/requirements accordingly.

## Orchestrator decomposition details

- [ ] Target module split: `graph_loader.py` (load/normalize config, handle env substitution), `graph_validator.py` (schema + runtime checks like declared outputs vs mappings), `graph_builder.py` (construct DAG and topological order), `execution_engine.py` (node scheduling, retries, loop/parallel controls), `context_injection.py` (apply agent capability metadata to inputs), `artifact_writer.py` (pluggable output writers), `visualization.py` (pyvis rendering), `metrics.py` (structured events + aggregation), and a slim `orchestrator.py` facade that wires them together.
- [ ] Normalize agent interactions: define a `NodeContext` object passed into execute (run_id, node_id, attempt, project_base_output_dir, capability flags) and a `NodeResult` (data dict, errors, metrics, artefacts) to replace raw dict merging.
- [ ] Replace special-casing with capability registry: agents declare `requires_project_dir`, `consumes_all_outputs`, `needs_input_keys`, `optional_input_keys`, `loop_over`, `parallelizable`. The engine reads this and injects context, removes ad-hoc type checks, and can validate missing inputs before execution.
- [ ] Isolate persistence: artifact writers take a registry of `output_key -> writer` mappings plus config-defined folder names; writers can be reused by plugins without editing core engine.
- [ ] Unify logging/metrics hooks: emit events (`node_started`, `node_completed`, `node_failed`, `retry`, `skipped`, `visualized`) through a lightweight bus so CLI/GUI/tests can subscribe without touching core logic.
- [ ] Keep backwards compatibility: add an adapter layer that accepts the current graph definition shape and translates it into the new internal models so existing configs continue to run while the new structure lands.

## Options for a robust, generalised scheme

- Option A (event-driven): central event bus with subscribers for logging, metrics, visualization, and artifact writing; engine only publishes events. Pros: pluggable observers; Cons: more moving parts.
- Option B (service interfaces): define narrow interfaces (e.g., `GraphStore`, `Scheduler`, `ArtifactWriter`, `Visualizer`, `MetricsSink`) with default implementations; allow dependency injection for tests/plugins. Pros: clearer contracts; Cons: slightly more boilerplate.
- Option C (hybrid): keep service interfaces but surface events for observability-only concerns; best if we want both strict contracts and easy tap-in for UI/telemetry.
- Recommendation: Option C (hybrid). It matches MACS's multi-entrypoint setup (CLI/GUI/adaptive) by keeping strict service contracts while allowing lightweight observability hooks without entangling core execution.
- Scheduling choices: start with deterministic topological order; optional parallel execution per node with bounded workers; later add queue-based scheduler to support backpressure and prioritization without changing agent code.
