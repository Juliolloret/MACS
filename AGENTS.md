# MACS Agent-Based System

MACS (Multi-Agent Collaboration System) organises research workflows as a directed acyclic graph (DAG) of specialised agents. Each agent focuses on a single responsibility—loading data, performing an LLM call, persisting artefacts, or validating results—so that complex pipelines can be composed by connecting agents together. The workflow topology, prompts, and runtime parameters are all declared in `config.json`, allowing you to reconfigure or swap out agents without editing orchestration code.

## How agents fit into a workflow

- **Graph-driven execution.** The orchestrator in `multi_agent_llm_system.py` reads the `graph_definition` section of `config.json`, instantiates each node's `type`, and executes the nodes in topological order. Inputs produced by upstream agents are injected into the downstream nodes according to the edge mapping defined in the configuration.
- **Shared base class.** All built-in agents inherit from `agents.base_agent.Agent`, which resolves the model to call (`model_key`), the system message template (`system_message_key`), and exposes the `llm.complete()` helper used to call the configured LLM.
- **Standard data contract.** Every agent returns a dictionary. Successful results expose named fields (for example `summary`, `vector_store_path`, or `integrated_knowledge_brief`). Failures are communicated via an `error` key or a dedicated `<field>_error` flag so that downstream agents can react appropriately.
- **Configuration-driven prompts.** Many agents accept runtime knobs such as `temperature`, `reasoning_effort`, or output directories through their `config_params`. These values live alongside the prompts inside `config.json`, keeping behaviour reproducible across runs.
- **External dependencies.** Some agents rely on optional libraries: PDF ingestion uses `PyPDF2`, while vector-store agents depend on `langchain_community.vectorstores.FAISS`. When these packages are unavailable the agent returns an `error` entry instead of raising an unhandled exception.

## Agent quick reference

| Agent | Responsibility | Primary inputs | Primary outputs | Useful config keys |
| --- | --- | --- | --- | --- |
| `PDFLoaderAgent` | Extract raw text from an individual PDF file. | `pdf_path` | `pdf_text_content`, `original_pdf_path` | – |
| `PDFSummarizerAgent` | Summarise the text of a single PDF with an LLM. | `pdf_text_content`, `original_pdf_path` | `summary`, `original_pdf_path` | `max_input_length`, `temperature`, `reasoning_effort`, `verbosity` |
| `PDFSummaryWriterAgent` | Persist individual summaries to `.txt` files. | `summaries_to_write`, `project_base_output_dir` | `written_summary_files` | `output_dir` |
| `MultiDocSynthesizerAgent` | Combine multiple summaries into a cross-document synthesis. | `all_pdf_summaries` | `multi_doc_synthesis_output` | `max_combined_len`, `temperature` |
| `ShortTermMemoryAgent` | Build a temporary FAISS store for the current run. | `individual_summaries`, `project_base_output_dir` | `vector_store_path`, `individual_summaries` | `vector_store_path_key` |
| `LongTermMemoryAgent` | Maintain a persistent FAISS store that accumulates knowledge across runs. | `individual_summaries`, `project_base_output_dir` | `long_term_memory_path` | `storage_filename_key` |
| `DeepResearchSummarizerAgent` | Answer a query by searching a vector store and synthesising the results. | `user_query`, `vector_store_path` | `deep_research_summary` | `top_k`, `temperature`, `reasoning_effort`, `verbosity` |
| `WebResearcherAgent` | Produce additional context that mimics lightweight web research. | `cross_document_understanding` | `web_summary` | `temperature` |
| `ExperimentalDataLoaderAgent` | Load and optionally summarise experimental data from disk. | `experimental_data_file_path` | `experimental_data_summary` | `use_llm_to_summarize_data`, `max_exp_data_len`, `temperature`, `reasoning_effort`, `verbosity` |
| `KnowledgeIntegratorAgent` | Merge upstream artefacts into a comprehensive knowledge brief. | `multi_doc_synthesis`, `deep_research_summary`, `web_research_summary`, `experimental_data_summary`, `all_agent_outputs` | `integrated_knowledge_brief`, `knowledge_sections`, `contributing_agents`, `agent_context_details` | `max_input_segment_len`, `temperature`, `reasoning_effort`, `verbosity` |
| `HypothesisGeneratorAgent` | Generate structured hypotheses from the integrated brief. | `integrated_knowledge_brief` | `hypotheses_output_blob`, `hypotheses_list`, `key_opportunities` | `num_hypotheses`, `max_brief_len`, `temperature`, `reasoning_effort`, `verbosity` |
| `ExperimentDesignerAgent` | Draft detailed experiment plans for each hypothesis. | `hypothesis` *or* `hypotheses_list` | `experiment_design` *or* `experiment_designs_list` | `reasoning_effort`, `verbosity` |
| `ObserverAgent` | Audit workflow outputs and surface reported errors. | `outputs_history` | `errors_found`, `errors` | – |

## Detailed agent reference

### Document ingestion and summarisation

#### `PDFLoaderAgent`
- Opens PDFs with `PyPDF2.PdfReader`, decrypting the file when possible, and concatenates the extracted text from every page.
- Returns both the raw text (`pdf_text_content`) and the original path so downstream agents can attribute summaries to specific files.
- Emits an `error` field when the path is missing, the file cannot be decrypted, or no text can be extracted.

#### `PDFSummarizerAgent`
- Builds an LLM prompt that references the original filename and the PDF text (truncated to `max_input_length` characters when necessary).
- Accepts optional reasoning controls (`temperature`, `reasoning_effort`, `verbosity`) and forwards them to the configured LLM client.
- Propagates upstream errors (for example, missing text) to simplify debugging in multi-step workflows.

#### `PDFSummaryWriterAgent`
- Writes each summary to a deterministic filename derived from the source PDF and the configured `output_dir`, defaulting to `pdf_summaries/` inside the project output directory.
- Skips summaries that are empty and logs—but does not raise—`OSError` write failures so the workflow can continue processing other files.

### Memory agents

#### `ShortTermMemoryAgent`
- Collects summary dictionaries, filters out entries without usable `summary` text, and uses the LLM's embeddings client to create a FAISS index for the current run.
- Saves the index to `project_base_output_dir/<resolved_filename>`, where `<resolved_filename>` is looked up via `vector_store_path_key` in `system_variables`. Missing configuration results in a warning and a sensible default filename.
- Returns `vector_store_path` (or `None` if nothing could be embedded) together with the validated summaries so that downstream agents know which inputs were indexed.

#### `LongTermMemoryAgent`
- Resolves its storage path via `storage_filename_key` and loads the previous FAISS index when present, appending any new summaries provided in `individual_summaries`.
- When no new summaries are supplied, the agent simply reports the current long-term memory path, allowing the orchestrator to continue without modification.
- All FAISS operations are guarded so missing dependencies or persistence issues are returned as explicit `error` messages instead of raising exceptions.

### Cross-document research

#### `MultiDocSynthesizerAgent`
- Validates that `all_pdf_summaries` is a list of dictionaries, ignoring any entries that already contain an `error` flag.
- Concatenates the summaries with source attribution, truncating the combined prompt to `max_combined_len` characters before invoking the LLM.
- Returns the synthesised narrative under `multi_doc_synthesis_output` or an `error` message when no valid summaries are available.

#### `DeepResearchSummarizerAgent`
- Loads the FAISS store referenced by `vector_store_path`, runs a similarity search for the provided `user_query`, and assembles the retrieved passages into the LLM prompt.
- Adjustable parameters include the number of neighbours to retrieve (`top_k`) and optional reasoning controls forwarded to the LLM call.
- Provides descriptive error messages when queries are missing, the vector store cannot be loaded, or FAISS is not installed.

#### `WebResearcherAgent`
- Generates supplementary context by prompting the LLM with the cross-document understanding produced upstream.
- Returns an empty `web_summary` when no input context is available, enabling optional use in the workflow without special branching.

### Data integration and reasoning

#### `ExperimentalDataLoaderAgent`
- Reads the file referenced by `experimental_data_file_path`, returning human-readable error text if the path is missing or the file cannot be found.
- When `use_llm_to_summarize_data` is enabled, truncates large files to `max_exp_data_len` characters before asking the LLM to produce a summary. If the LLM call fails it falls back to returning the raw content along with a warning.
- Always emits an `experimental_data_summary`, ensuring downstream integrators have something to work with even when the source file is empty or missing.

#### `KnowledgeIntegratorAgent`
- Builds a structured prompt from every upstream contribution—cross-document synthesis, deep research, web findings, experimental data, and any additional agent outputs—while tracking which agents supplied usable content.
- Applies `max_input_segment_len` truncation per section, includes upstream error details when present, and returns both the integrated brief and a machine-readable breakdown via `knowledge_sections` and `agent_context_details`.
- Forwards reasoning controls (`temperature`, `reasoning_effort`, `verbosity`) to the final LLM call that produces `integrated_knowledge_brief`.

### Hypothesis generation and experiment design

#### `HypothesisGeneratorAgent`
- Validates and formats the requested `num_hypotheses`, injects it into the system prompt, and truncates overly long knowledge briefs before calling the LLM.
- Parses the LLM response as JSON, returning structured entries under `hypotheses_list` alongside the raw blob and any `key_opportunities` narrative. Parsing errors are logged and surfaced via an `error` field.

#### `ExperimentDesignerAgent`
- Works with either single hypotheses (via the `hypothesis` field) or batches (`hypotheses_list`), returning matching output keys so the orchestrator can branch accordingly.
- Skips invalid hypothesis objects, reports the issue in the returned dictionary, and otherwise generates detailed experiment designs using the configured LLM and optional reasoning controls.

### Observability

#### `ObserverAgent`
- Inspects the `outputs_history` dictionary generated by the orchestrator and collates any `error` entries reported by previous agents.
- Returns a compact summary (`errors_found` plus a map of agent IDs to error messages) that can be surfaced in the GUI or logs.

## Extending with new agents

MACS can be extended without modifying the orchestrator:

1. **Create an agent class.** Place the implementation inside `agents/` (for built-ins) or `agent_plugins/` (for external plugins). Decorate built-in agents with `@register_agent("YourAgentName")` so they are picked up when `agents.load_agents()` runs.
2. **Inherit from `Agent`.** Implement the `execute(self, inputs: dict) -> dict` method and follow the dictionary contract for outputs and errors. You get access to the configured LLM client, system prompt, and helper logging utilities from the base class.
3. **Provide plugin metadata (optional).** To ship an agent as a plugin, expose a module-level `PLUGIN` variable of type `AgentPlugin`. MACS validates the declared `macs_version` for compatibility before registering the plugin.
4. **Reference the agent in `config.json`.** Add a node to the `graph_definition` using the agent name (built-in or plugin) and map its inputs/outputs to neighbouring nodes.
5. **Verify registration.** Run `python cli.py --list-plugins` to inspect all available agents, including dynamically loaded plugins. The command prints every registered agent name and version so you can confirm discovery before executing a workflow.

Example plugin (`agent_plugins/my_new_agent.py`):

```python
from agents.base_agent import Agent
from agents.registry import AgentPlugin, PluginMetadata, MACS_VERSION

class MyNewAgent(Agent):
    def execute(self, inputs: dict) -> dict:
        # Your agent's logic here
        return {"output_key": "some_value"}

PLUGIN = AgentPlugin(
    agent_class=MyNewAgent,
    metadata=PluginMetadata(
        name="my_new_agent",
        version="0.1.0",
        author="Your Name",
        macs_version=MACS_VERSION,
    ),
)
```

Once the file is in place the system automatically discovers and registers it. Add the agent's `name` to your workflow configuration to begin using it immediately.
