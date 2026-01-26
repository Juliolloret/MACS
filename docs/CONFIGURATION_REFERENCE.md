# MACS Configuration Reference

This document explains the structure of `config.json`. The goal is to help you
edit the workflow safely, even if you are new to configuration files.

## 1. What Is `config.json`?

- It is a JSON file (a text file with key-value pairs).
- MACS reads it at startup to decide which agents to run and in what order.
- Both the GUI and CLI use the same configuration.

If the file is malformed, MACS will print a clear error and stop.

## 2. Minimal Example (Shape Only)

This is a simplified version of the real config. It shows the structure, not
all possible settings:

```json
{
  "system_variables": {
    "openai_api_key": "sk-...",
    "default_llm_model": "gpt-4.1-mini",
    "output_project_synthesis_folder_name": "project_synthesis",
    "short_term_memory_filename": "faiss_index_stm",
    "long_term_memory_filename": "faiss_index_ltm"
  },
  "agent_prompts": {
    "pdf_summarizer_sm": "Summarize the following PDF..."
  },
  "graph_definition": {
    "nodes": [
      {
        "id": "pdf_loader_node",
        "type": "PDFLoaderAgent",
        "config": {"description": "Load PDFs"}
      },
      {
        "id": "pdf_summarizer_node",
        "type": "PDFSummarizerAgent",
        "config": {
          "system_message_key": "pdf_summarizer_sm",
          "loop_over": "loader_results"
        }
      }
    ],
    "edges": [
      {
        "from": "pdf_loader_node",
        "to": "pdf_summarizer_node",
        "data_mapping": {"results": "loader_results"}
      }
    ]
  }
}
```

## 3. `system_variables`

This section controls global behavior:

- `openai_api_key` - required for real LLM calls.
- `llm_client` - use `openai` for the real API, otherwise the fake client.
- `default_llm_model` - fallback model for any agent that does not specify one.
- `models` - optional map of agent model keys to concrete model names.
- `prompt_ids` - optional list of OpenAI Responses API prompt IDs.
- `output_project_*_folder_name` - names of output folders.
- `short_term_memory_filename` and `long_term_memory_filename` - FAISS index
  file names for memory agents.

Tip: Keep secrets out of Git by using environment variables when possible.

## 4. `agent_prompts`

This is a dictionary of reusable system prompts. Agents reference prompts by
`system_message_key` in their node config. Example:

```json
"agent_prompts": {
  "knowledge_integrator_sm": "Combine all inputs into a clear brief..."
}
```

If a key is missing, the orchestrator logs a warning and uses an empty prompt.

## 5. `graph_definition`

The graph definition is the heart of MACS. It has two parts: `nodes` and
`edges`.

### 5.1 Nodes

Each node is one agent instance.

Required fields:

- `id` - unique string name for this node.
- `type` - the agent class name (must be registered).

Optional `config` fields include:

- `description` - human-readable label for logs.
- `model_key` - key used to look up a model in `system_variables.models`.
- `system_message_key` - prompt key from `agent_prompts`.
- `loop_over` - name of a list input to iterate over.
- `parallel_execution` and `max_workers` - allow parallel loops.
- `temperature`, `reasoning_effort`, `max_input_length`, `top_k` - agent
  parameters forwarded to the agent.
- `failure_policy` and `retries` - error handling rules.

### 5.2 Edges

Edges define how outputs flow between nodes. Example:

```json
{
  "from": "pdf_summarizer_node",
  "to": "multi_doc_synthesizer",
  "data_mapping": {"results": "all_pdf_summaries"}
}
```

This means: take `results` from the summarizer output and pass it as
`all_pdf_summaries` to the synthesizer.

If a key is missing at runtime, MACS records an error and may skip the node.

## 6. How to Add a New Node (Beginner Steps)

1. Decide which agent class you want to use (from `agents/` or a plugin).
2. Add a node to `graph_definition.nodes` with a unique `id`.
3. Add an edge from the upstream node that provides the input you need.
4. Run `python -c "from utils import load_app_config; load_app_config('config.json')"`
   to validate the file.
5. Run a small test (a few PDFs) to confirm the new node works.

## 7. Validation and Troubleshooting

- Syntax errors: Use a JSON formatter or `python -m json.tool config.json`.
- Missing node IDs: check typos in `edges`.
- Wrong data_mapping: print upstream outputs or inspect `node_metrics.json`.
- If unsure, run tests: `python scripts/run_tests.py`.

For deeper context, see `docs/DEVELOPMENT_GUIDE.md`.
