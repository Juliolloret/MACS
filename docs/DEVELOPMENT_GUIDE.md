# MACS Development Guide

This guide provides comprehensive instructions for developers—human or AI—who wish to extend the **Multi-Agent Collaboration System (MACS)**. It covers project structure, setup, conventions, and typical workflows so that contributors can quickly become productive.

---

## 1. Repository Structure

The core components of MACS are organized as follows:

| Path | Description |
| ---- | ----------- |
| `agents/` | Individual agent implementations. Each file subclasses `Agent` from `agents.base_agent` and registers itself with `@register_agent`. |
| `agent_plugins/` | Optional drop-in agents that can be discovered at runtime. Useful for experiments or external contributions. |
| `adaptive/` | Components for evolutionary runs where MACS evaluates results and mutates its own configuration for the next iteration. |
| `docs/` | Project documentation (including this guide). |
| `tests/` | Pytest-based test suite covering agents, orchestrator behaviour, and integration flows. |
| `cli_test.py`, `gui.py` | Entry points for command-line and graphical execution. |
| `config.json` | Default configuration and agent graph used by `GraphOrchestrator`. |
| `utils.py`, `memory.md`, etc. | Supporting utilities and memory model description. |

When adding new modules, strive to keep the hierarchy shallow and file names descriptive.

---

## 2. Environment Setup

1. **Python Version** – MACS targets Python **3.10+**.
2. **Dependencies** – Install required packages:
   ```bash
   pip install -r requirements.txt
   ```
3. **LLM Credentials** – Some agents rely on external LLM providers such as OpenAI. Set environment variables (e.g. `OPENAI_API_KEY`, `OPENAI_PROMPT_ID`, `OPENAI_CONVERSATION_ID`) before running tests or the CLI.
4. **Optional Tools** – For development you may also install linters like `ruff` or formatters like `black`.

---

## 3. Running MACS

### Command-Line
```
python cli_test.py --pdf-dir path/to/pdfs --output-dir run_output
```

### Graphical Interface
```
python gui.py
```

Both frontends load `config.json` and delegate execution to `GraphOrchestrator` in `multi_agent_llm_system.py`.

---

## 4. Configuration and Graphs

The orchestrator executes agents according to a directed acyclic graph described in `config.json` (or an alternative config file). Important fields include:

- `llm` – Model selection and API parameters.
- `graph_definition` – Nodes (agents) and edges (data flow).
- `prompts` – Template strings referenced by agents.

To modify the workflow:

1. Add or adjust nodes in `graph_definition`.
2. Ensure each node name matches an agent `name` attribute.
3. Update edges to reflect data dependencies.

For complex experiments, create a new configuration file and pass it to the CLI/GUI via `--config`.

See [CONFIGURATION_REFERENCE.md](CONFIGURATION_REFERENCE.md) for a thorough breakdown of every configuration section, common parameters, and validation tips.

---

## 5. Creating or Modifying Agents

Agents encapsulate a single capability—loading PDFs, summarizing, generating hypotheses, etc. To implement a new agent:

1. **Import** `Agent` from `agents.base_agent` and inherit from it. The orchestrator supplies the `llm`, `config_params`, and `app_config` arguments expected by the base class constructor.
2. **Implement** the `execute(self, inputs: dict) -> dict` method. Use `self.llm` for language-model calls and read configuration from `self.config_params`.
3. **Return** a dictionary describing the results (strings, lists, nested dicts, and optional `error` keys). Downstream nodes receive these values as inputs.
4. **Register** the class with `@register_agent("YourAgentName")` from `agents.registry` so the orchestrator can discover it by name.
5. **Place** the implementation in `agents/` for built-ins or `agent_plugins/` for optional plugins, and reference the chosen agent name inside the configuration graph.

When adapting existing agents, maintain backwards compatibility with their inputs/outputs unless the configuration is updated accordingly.

---

## 6. Memory System

MACS uses both short-term and long-term memory agents:

- **ShortTermMemoryAgent** – Stores information for the current run.
- **LongTermMemoryAgent** – Persists data between runs (e.g., on disk).

Memory agents implement `load`, `save`, and retrieval methods. When introducing new memory features, document the data schema and storage location to keep the system interoperable.

---

## 7. Adaptive Mode

The optional `adaptive` package allows MACS to evaluate its own outputs and iterate. Key components:

1. `adaptive_graph_runner.py` – Runs the configured graph and gathers metrics.
2. Mutation functions in `adaptive.mutation` – Alter the configuration (adding/removing agents, adjusting prompts).
3. Selection mechanisms – Determine whether to keep the new configuration based on evaluation score.

To experiment with adaptive runs, execute:
```bash
python adaptive/adaptive_graph_runner.py --config config.json
```

### Evaluation Plugins

Adaptive runs can use custom evaluation plugins to score graph outputs. An
evaluation plugin implements an ``evaluate(outputs, graph_def, threshold, step)``
method and returns a numeric score. Plugins can be referenced in
``config.json`` using ``evaluation_plugins`` with entries like
``"agent_plugins.sample_evaluator:SampleEvaluator"`` or exposed via the
``macs.evaluators`` entry point. When multiple plugins are provided, their
scores are averaged to decide whether the graph should mutate further.

---

## 8. Coding Style

- Follow **PEP8** guidelines.
- Prefer descriptive variable names and type hints (`from __future__ import annotations` if needed).
- Write docstrings for all public functions and classes.
- Keep functions small and focused; factor out helpers where useful.
- Avoid hard-coding paths; accept them as parameters or use configuration values.

---

## 9. Testing

1. Run the full test suite before submitting changes:
   ```bash
   pytest
   ```
2. Add unit tests for new agents or utilities.
3. Tests are organized roughly by agent; replicate existing patterns when adding new tests.
4. Use fixtures to mock external services (e.g., LLM calls) where possible to keep tests deterministic.

---

## 10. Documentation Practices

- Place new design docs in `docs/` with clear, descriptive names.
- Cross-reference related documents using Markdown links.
- Keep diagrams in `docs/` or referenced via relative paths.
- Update `README.md` when introducing major features.

---

## 11. Working with AI Co-Developers

MACS is designed for collaboration between humans and AI:

- **Commit Messages** – Use clear, imperative summaries so agents can infer intent.
- **Atomic Changes** – Keep pull requests focused; avoid mixing unrelated changes.
- **Reasoning Artifacts** – When an AI contributes, capture its rationale in comments or additional docs so future agents understand previous decisions.
- **Feedback Cycle** – Leverage code review to refine prompts and agent behaviour iteratively.

---

## 12. Troubleshooting

- **Missing API Keys** – If LLM-dependent tests fail, ensure the appropriate environment variables are set.
- **Import Errors** – Verify new modules are included in `__init__.py` files if necessary.
- **Graph Deadlocks** – The orchestrator requires a DAG; check for cycles or missing edges if execution halts.
- **File Paths** – Use absolute or configuration-based paths when running inside different environments (e.g., containers vs. local).

---

## 13. Getting Help

- Review existing documentation in `docs/` and `README.md`.
- Check open issues and discussions in the repository for known problems.
- When in doubt, open a GitHub issue or start a discussion thread.

---

With this guide, contributors—whether human developers or automated agents—should be able to understand the MACS architecture and extend it confidently.
