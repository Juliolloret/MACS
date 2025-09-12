# MACS Agent-Based System

MACS (Multi-Agent Collaboration System) uses a modular, agent-based architecture to perform complex research workflows. The system is designed as a directed acyclic graph (DAG) of agents, where each agent is a specialized worker that performs a specific task. The flow of data and the execution order are defined in a central configuration file (`config.json`).

This approach allows for flexible and customizable workflows. You can reconfigure the graph, change agent parameters, or even swap out agents with different implementations without altering the core orchestration logic.

## Core Agents

The core agents of the MACS system are located in the `agents/` directory. Below is a list of the available agents and their functions.

| Agent Class | Description | Inputs | Outputs |
| --- | --- | --- | --- |
| `PDFLoaderAgent` | Extracts text content from all PDF files in a specified directory. | `pdf_folder` (path) | `pdf_texts` (list of dicts with `path` and `text`) |
| `PDFSummarizerAgent` | Takes the text of a single PDF and uses an LLM to generate a concise summary. | `pdf_text` (string) | `summary` (string) |
| `PDFSummaryWriterAgent`| Writes individual PDF summaries to separate text files in the output directory. | `summaries_to_write` (list of dicts) | `written_summary_files` (list of paths) |
| `MultiDocSynthesizerAgent`| Combines a list of individual document summaries into a single, coherent synthesis. | `all_pdf_summaries` (list of dicts) | `multi_doc_synthesis_output` (string) |
| `ShortTermMemoryAgent`| Creates a temporary FAISS vector store from document summaries for semantic search within the current session. | `individual_summaries` (list of strings) | `vector_store_path` (path) |
| `LongTermMemoryAgent` | Manages a persistent FAISS vector store, accumulating knowledge across different sessions by adding new summaries. | `individual_summaries` (list of strings) | `long_term_memory_path` (path) |
| `DeepResearchSummarizerAgent`| Performs a semantic search on a vector store using a user query and synthesizes a summary from the retrieved results. | `user_query` (string), `vector_store_path` (path) | `deep_research_summary` (string) |
| `WebResearcherAgent` | Uses an LLM to simulate a web search based on a cross-document summary to add additional context. | `cross_document_understanding` (string) | `web_summary` (string) |
| `ExperimentalDataLoaderAgent`| Loads and structures experimental data from a text file. | `experimental_data_path` (path) | `experimental_data` (structured object) |
| `KnowledgeIntegratorAgent`| Merges outputs from various agents (e.g., deep summary, long-term memory, experimental data) into a final "knowledge brief". | `deep_research_summary`, `ltm_query_result`, `experimental_data` | `integrated_knowledge_brief` (string) |
| `HypothesisGeneratorAgent`| Generates novel, testable research hypotheses based on the integrated knowledge brief. | `knowledge_brief` (string) | `hypotheses` (list of strings) |
| `ExperimentDesignerAgent`| Outlines a detailed experimental plan to test a given hypothesis. | `hypothesis` (string) | `experiment_design` (string) |
| `ObserverAgent` | Reviews the outputs of all other agents to detect and report errors or inconsistencies. | `all_agent_outputs` (dict) | `observations` (string) |

## Extending with New Agents

MACS can be extended with new agents through a plugin system. To add a new agent, follow these steps:

1.  **Create a Python file** in the `agent_plugins/` directory.
2.  **Define your agent class**, inheriting from `agents.base_agent.Agent`.
3.  **Implement the `execute` method** for your agent's logic.
4.  **Register your agent** by creating a `PLUGIN` variable of type `AgentPlugin`.

Example (`agent_plugins/my_new_agent.py`):

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

Once the file is in the `agent_plugins/` directory, the system will automatically discover and register it. You can then add it to your workflow by referencing its `name` in `config.json`.

You can list all registered agents, including plugins, by running:
```bash
python cli.py --list-plugins
```
