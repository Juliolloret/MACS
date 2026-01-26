# MACS Agent-Based System

MACS (Multi-Agent Collaboration System) organizes research workflows as a
Directed Acyclic Graph (DAG) of specialized agents. Each agent has one clear
job: load data, call an LLM, write files, or validate results. This makes the
system easier to extend and debug.

## What Is an Agent? (Simple Explanation)

An agent is just a Python class with one method:

- `execute(self, inputs: dict) -> dict`

It receives a dictionary, does work, and returns a dictionary.

## How Agents Fit into a Workflow

- The orchestrator reads `graph_definition` from `config.json`.
- Each node in the graph points to an agent class by name.
- Edges map output keys from one agent into input keys for the next.
- Agents run in topological order so inputs are ready when needed.

## Agent Capabilities System

Agents declare their requirements using class-level attributes:

```python
class MyAgent(Agent):
    REQUIRES_PROJECT_DIR = True           # Needs project_base_output_dir
    CONSUMES_ALL_OUTPUTS = False          # Needs all_agent_outputs
    REQUIRED_INPUTS = ["pdf_text_content"]  # Must be provided
    OPTIONAL_INPUTS = ["user_query"]       # May be provided
```

The orchestrator uses `agent.get_capabilities()` to:
- Validate required inputs are mapped in the graph
- Inject project directory for agents that write files
- Aggregate all outputs for integrator agents

The `AgentCapabilities` dataclass (in `contracts.py`) includes:
- `requires_project_dir`: agent writes to the output directory
- `consumes_all_outputs`: agent needs all upstream results
- `needs_input_keys`: list of required inputs
- `optional_input_keys`: list of optional inputs
- `loop_over`: key to iterate over (for loop execution)
- `parallelizable`: can run iterations in parallel

## Input and Output Contract (Example)

Example of a simple agent output:

```python
{
  "summary": "Short summary text",
  "original_pdf_path": "papers/example.pdf"
}
```

If something goes wrong, agents return an `error` key:

```python
{
  "error": "PDF could not be read"
}
```

Downstream agents can check for `error` and decide whether to skip or continue.

## Agent Quick Reference

| Agent | Responsibility | Primary inputs | Primary outputs | Useful config keys |
| --- | --- | --- | --- | --- |
| `PDFLoaderAgent` | Extract raw text from an individual PDF file. | `pdf_path` | `pdf_text_content`, `original_pdf_path` | - |
| `PDFSummarizerAgent` | Summarize the text of a single PDF with an LLM. | `pdf_text_content`, `original_pdf_path` | `summary`, `original_pdf_path` | `max_input_length`, `temperature`, `reasoning_effort`, `verbosity` |
| `PDFSummaryWriterAgent` | Persist individual summaries to `.txt` files. | `summaries_to_write`, `project_base_output_dir` | `written_summary_files` | `output_dir` |
| `MultiDocSynthesizerAgent` | Combine multiple summaries into a cross-document synthesis. | `all_pdf_summaries` | `multi_doc_synthesis_output` | `max_combined_len`, `temperature` |
| `ShortTermMemoryAgent` | Build a temporary FAISS store for the current run. | `individual_summaries`, `project_base_output_dir` | `vector_store_path`, `individual_summaries` | `vector_store_path_key` |
| `LongTermMemoryAgent` | Maintain a persistent FAISS store across runs. | `individual_summaries`, `project_base_output_dir` | `long_term_memory_path` | `storage_filename_key` |
| `DeepResearchSummarizerAgent` | Answer a query by searching a vector store. | `user_query`, `vector_store_path` | `deep_research_summary` | `top_k`, `temperature`, `reasoning_effort`, `verbosity` |
| `WebResearcherAgent` | Produce additional context that mimics lightweight web research. | `cross_document_understanding` | `web_summary` | `temperature` |
| `ExperimentalDataLoaderAgent` | Load and optionally summarize experimental data from disk. | `experimental_data_file_path` | `experimental_data_summary` | `use_llm_to_summarize_data`, `max_exp_data_len`, `temperature`, `reasoning_effort`, `verbosity` |
| `KnowledgeIntegratorAgent` | Merge upstream artifacts into a knowledge brief. | `multi_doc_synthesis`, `deep_research_summary`, `web_research_summary`, `experimental_data_summary`, `all_agent_outputs` | `integrated_knowledge_brief`, `knowledge_sections`, `contributing_agents`, `agent_context_details` | `max_input_segment_len`, `temperature`, `reasoning_effort`, `verbosity` |
| `HypothesisGeneratorAgent` | Generate structured hypotheses from the integrated brief. | `integrated_knowledge_brief` | `hypotheses_output_blob`, `hypotheses_list`, `key_opportunities` | `num_hypotheses`, `max_brief_len`, `temperature`, `reasoning_effort`, `verbosity` |
| `ExperimentDesignerAgent` | Draft experiment plans for each hypothesis. | `hypothesis` or `hypotheses_list` | `experiment_design` or `experiment_designs_list` | `reasoning_effort`, `verbosity` |
| `ObserverAgent` | Audit workflow outputs and surface reported errors. | `outputs_history` | `errors_found`, `errors` | - |

## Detailed Agent Reference

### Document Ingestion and Summarization

#### `PDFLoaderAgent`
- Opens PDFs with `pypdf.PdfReader`.
- Returns raw text plus the original PDF path.
- Returns `error` when the file is missing, encrypted, or empty.

#### `PDFSummarizerAgent`
- Builds an LLM prompt with the PDF text and filename.
- Truncates long text to `max_input_length`.
- Passes `temperature`, `reasoning_effort`, and `verbosity` to the LLM client.

#### `PDFSummaryWriterAgent`
- Writes summaries to `pdf_summaries/` inside the project output dir.
- Skips empty summaries and logs file write errors instead of crashing.

### Memory Agents

#### `ShortTermMemoryAgent`
- Builds a FAISS index for the current run.
- Saves it to `project_base_output_dir/<short_term_memory_filename>`.
- Returns the path for downstream agents.

#### `LongTermMemoryAgent`
- Loads the existing FAISS index (if any).
- Appends new summaries and saves back to disk.

### Cross-Document Research

#### `MultiDocSynthesizerAgent`
- Combines multiple summaries into one synthesis.
- Skips entries that already have `error`.

#### `DeepResearchSummarizerAgent`
- Loads a FAISS index and runs a similarity search.
- Builds a prompt from the top-k results.

#### `WebResearcherAgent`
- Generates extra context from the synthesis.
- Returns an empty `web_summary` if no context is available.

### Data Integration and Reasoning

#### `ExperimentalDataLoaderAgent`
- Reads a text file with experimental data.
- Can optionally summarize large files with the LLM.

#### `KnowledgeIntegratorAgent`
- Merges synthesis, deep research, web summary, and experimental data.
- Tracks which agents contributed content.

### Hypothesis and Experiment Design

#### `HypothesisGeneratorAgent`
- Asks the LLM for structured hypotheses in JSON.
- Returns parsed `hypotheses_list`.

#### `ExperimentDesignerAgent`
- Accepts a single hypothesis or a list.
- Skips invalid hypothesis objects and reports the issue in output.

### Observability

#### `ObserverAgent`
- Scans all outputs and collects errors.
- Returns a compact error map for the GUI or CLI.

## Adding a New Agent (Beginner Steps)

1. Create a file in `agents/` (for built-in) or `agent_plugins/` (for plugin).
2. Subclass `Agent` and implement `execute(self, inputs)`.
3. Register with `@register_agent("YourAgentName")`.
4. Add a node in `config.json` using the same name.
5. Add tests in `tests/` to verify the behavior.

Example plugin (`agent_plugins/my_new_agent.py`):

```python
from agents.base_agent import Agent
from agents.registry import AgentPlugin, PluginMetadata, MACS_VERSION

class MyNewAgent(Agent):
    def execute(self, inputs: dict) -> dict:
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

## Common Mistakes and Fixes

- Missing key in `data_mapping` -> check upstream output keys.
- Returning a non-dict from `execute` -> always return a dictionary.
- Forgetting to register the agent -> use `@register_agent` or plugin metadata.
- Using a prompt key that does not exist -> update `agent_prompts` in config.
