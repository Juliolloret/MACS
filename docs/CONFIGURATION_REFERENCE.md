# MACS Configuration Reference

This document explains how to customize the `config.json` file that drives the
MACS orchestration graph. Use it alongside the high-level overview in
[README.md](../README.md) and the development notes in
[docs/DEVELOPMENT_GUIDE.md](DEVELOPMENT_GUIDE.md).

The configuration file is loaded by `utils.load_app_config()` and validated
against `config_schema.py`. Both the command-line interface (`cli.py`) and the
GUI (`gui.py`) rely on the same structure, so changes immediately affect every
entry point. When the adaptive runner mutates a graph, it stores successive
versions under `adaptive/config_history/` so you can audit how the pipeline
changed over time.

---

## 1. File Layout and Loading

* **Default file** – `config.json` in the repository root. Supply a different
  path with `python cli.py --config custom.json` or via the GUI.
* **Runtime validation** – `load_app_config()` parses the file, validates the
  `graph_definition` with Pydantic models from `config_schema.py`, and caches
  the result in `utils.APP_CONFIG` so helper functions like `get_model_name()`
  and `get_prompt_text()` can resolve values consistently.
* **Run history** – Each orchestration run records the resolved configuration
  in `storage/run_configs/<run_id>.json`. Use `storage.run_history.list_runs()`
  to retrieve previous configurations for reproducibility.

---

## 2. `system_variables`

The `system_variables` section configures global behaviour such as API
credentials, model selection, caching, and output paths. Common keys include:

| Key | Description |
| --- | --- |
| `openai_api_key` | API key used by the OpenAI-backed LLM client. Required unless you run with the fake client. |
| `llm_client` | Chooses the backend (`"openai"` for the SDK, anything else falls back to `FakeLLM`). |
| `openai_api_timeout_seconds` | Timeout applied to OpenAI API calls. Defaults to `120` seconds. |
| `cache_dir` | Directory used by `llm_openai.OpenAILLM` to persist response and embedding caches. |
| `default_llm_model` | Fallback model name used when an agent does not specify `model_key`. |
| `models` | Mapping of logical names (e.g. `"pdf_summarizer"`) to concrete model identifiers. Helpers like `get_model_name()` look up entries here. |
| `prompt_ids` | Optional list of prompt identifiers forwarded to the Responses API and captured in run history for auditing. |
| `output_project_*_folder_name` | Names of subdirectories created under the project output directory (e.g. `output_project_synthesis_folder_name`). |
| `short_term_memory_filename`, `long_term_memory_filename` | Filenames used by memory agents when writing FAISS indexes. |

### Model Overrides

Agents request models by passing `model_key` in their node configuration. The
key is resolved through `system_variables.models`; if no match exists, the
`default_llm_model` value is used. This pattern allows you to point specific
agents at lighter-weight models without editing code.

```json
{
  "system_variables": {
    "default_llm_model": "gpt-4.1-mini",
    "models": {
      "pdf_summarizer": "gpt-4.1-mini",
      "knowledge_integrator_model": "gpt-4.1"
    }
  }
}
```

### Output Management

`GraphOrchestrator._save_consolidated_outputs()` creates per-artifact folders
based on keys that start with `output_project_` and end with `_folder_name`.
For example, `output_project_synthesis_folder_name` becomes the `synthesis`
folder containing the cross-document summary, web research notes, and optional
experimental data summary. Adjust these names to reorganize exported files
without changing agent logic.

---

## 3. `agent_prompts`

This section stores reusable system prompts. Agents reference individual
entries via the `system_message_key` property in their node configuration. The
same key can be shared by multiple agents. Prompts are ordinary strings, so you
can add formatting cues, JSON output contracts, or reasoning hints as needed.

When a prompt key is missing or set to `null`, the orchestrator logs an error
and the agent receives an empty string. Keeping prompts centralized prevents
accidental drift between CLI and GUI runs.

---

## 4. `graph_definition`

The `graph_definition` describes the directed acyclic graph that connects all
agents. It contains two arrays: `nodes` and `edges`.

### 4.1 Nodes

Each entry in `nodes` must include an `id`, a `type` that matches the agent’s
registered name, and an optional `config` dictionary. Common `config`
parameters include:

| Parameter | Purpose |
| --- | --- |
| `description` | Human-readable explanation surfaced in logs. |
| `loop_over` | Name of the upstream list to iterate over. The orchestrator will call `execute()` once per element and collect the results under `results`. |
| `loop_item_input_key` | When looping, specifies the argument name that receives the current item. Defaults to `item`. |
| `parallel_execution` / `max_workers` | Enables threaded execution for looped agents. Useful for per-PDF summarization. |
| `model_key` | Logical model identifier resolved through `system_variables.models`. |
| `system_message_key` | Prompt key from `agent_prompts`. |
| `temperature`, `reasoning_effort`, `top_k`, `max_input_length`, etc. | Agent-specific knobs forwarded to `execute()`. Check the corresponding agent class for supported keys. |
| `failure_policy` | Overrides the orchestrator’s global failure handling. Supported values are `"continue"`, `"retry"`, and `"abort"`. |
| `retries` | Number of retry attempts when `failure_policy` is `"retry"`. |
| `allow_execution_with_errors` | Permit downstream execution even when upstream inputs flagged errors (used by `KnowledgeIntegratorAgent`). |

Specialised agents may expect additional parameters—for example, the memory
agents look for the filename keys described in §2, and the experimental data
loader can toggle `use_llm_to_summarize_data`.

### 4.2 Edges

Edges connect the output of one node to the input of another. Each edge record
specifies:

```json
{
  "from": "pdf_summarizer_node",
  "to": "multi_doc_synthesizer",
  "data_mapping": {
    "results": "all_pdf_summaries"
  }
}
```

`data_mapping` translates keys in the upstream output dictionary (`results`) to
parameter names expected by the downstream agent (`all_pdf_summaries`). If a key
is missing at runtime, the orchestrator records the issue and either injects a
placeholder or skips the node depending on `allow_execution_with_errors`.

`config_schema.GraphDefinition` verifies that every edge references valid node
IDs. Run `python -c "from utils import load_app_config; load_app_config('config.json')"`
after editing to catch validation errors early.

---

## 5. Optional Top-Level Sections

Beyond the three core sections, the configuration file can include:

| Section | Description |
| --- | --- |
| `evaluation_plugins` | A list of dotted paths resolved by `adaptive.evaluation.load_evaluation_plugins()`. Each plugin must expose an `evaluate(outputs, graph_def, threshold, step)` method returning a numeric score used during adaptive runs. |
| `experiment_optimization` | Settings consumed by adaptive tooling (e.g. mutation strategy, target metric names, or a SQLite database path for experiment records). See `experiment_config_example.json` for a template. |
| `project_metadata` (optional) | Arbitrary metadata persisted alongside run history records. |

These sections are ignored by the standard orchestrator but become important
when experimenting with the adaptive cycle in `adaptive/adaptive_graph_runner.py`.

---

## 6. Testing Your Changes

1. **Validate JSON structure** – Ensure the file remains well-formed. Most
   editors can format or lint JSON automatically.
2. **Run the loader** – Execute `python -c "from utils import load_app_config; load_app_config('config.json')"` to trigger schema
   validation and confirm prompt/model lookups succeed.
3. **Dry-run the CLI** – Use a small PDF directory and the `--out-dir` flag to
   confirm new folders, loops, or agents behave as expected. The orchestrator
   writes metrics to `node_metrics.json` in the project output directory and
   logs placeholder files when upstream errors occur.

With these guardrails, you can iteratively evolve the MACS workflow—adding new
agents, swapping models, or introducing evaluation plugins—without touching the
core Python code.
