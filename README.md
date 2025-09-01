<p align="center">
  <img src="https://github.com/user-attachments/assets/e2d9cd31-4e54-41f9-81d6-11304b0f80c1" alt="MACS Logo" width="250"/>
</p>

![GitHub Workflow Status](https://img.shields.io/github/workflow/status/Juliolloret/MACS/CI)
![License](https://img.shields.io/github/license/Juliolloret/MACS)
![Python Version](https://img.shields.io/badge/python-3.8+-blue)

---

# MACS: Multi-Agent LLM Research Assistant
**MACS** (Multi-Agent Collaboration System) is a open-source modular Python-based platform for orchestrating advanced research workflows on collections of academic PDFs. It leverages multiple intelligent agents and LLMs to extract, summarize, and synthesize information from academic PDFs and experimental data, supporting literature reviews, hypothesis generation, and experiment planning.

# Features
- Automated Literature Review: Extracts and summarizes research articles from PDFs.
- Multi-Document Synthesis: Integrates knowledge across multiple sources.
- Experimental Planning: Generates hypotheses and designs experiments automatically.
- User-Friendly GUI: Easily configure and launch workflows with real-time monitoring.
- Customizable Workflows: Adjust agent prompts, models, and workflow topology via config.json.
- Graph Visualization: Export the agent workflow graph to aid debugging and understanding.
- Extensible Architecture: Add new agents for specialized tasks.

## Table of Contents

- [Overview](#overview)
- [Components](#components)
- [Setup & Usage](#setup--usage)
- [Workflow Details](#workflow-details)
- [Run History and Reproducibility](#run-history-and-reproducibility)
- [Configuration](#configuration)
- [Extending & Contributing](#extending--contributing)
- [Credits](#credits)
---

## Components

### 1. `gui.py` — Graphical User Interface

- **Purpose:** User-friendly interface for setting up, launching, and monitoring integrated research analysis workflows.
- **Main Features:**
  - Select project name, input PDF folder(s), optional experimental data, output directory, and config file.
  - Initiate analysis via the "Start Integrated Analysis" button.
  - Monitor progress, logs, and errors in real time.
  - Handles backend import and configuration errors gracefully.
- **How it works:** The GUI interacts with the backend (`multi_agent_llm_system.py`), passing user selections and displaying status updates.

 <img src="https://github.com/user-attachments/assets/5c8fabc4-934f-42e9-849d-04b9412ed3b1" width="600" height="475"/>

### 2. `multi_agent_llm_system.py` — Multi-Agent Orchestrator

- **Purpose:** Core backend that loads and executes the multi-agent workflow defined in `config.json`.
- **Key Elements:**
  - **Graph Orchestrator:** The central component that builds a directed acyclic graph (DAG) of agents from the configuration file. It executes the agents in the correct topological order, managing the flow of data between them. It can also export the graph structure for visualization via Graphviz.
  - **Dynamic Agent Loading:** Agents are not hardcoded in the orchestrator. Instead, agent classes are dynamically loaded from the `agents` package based on the `type` specified for each node in the `config.json` graph.
  - **Agent Network:** The default workflow consists of specialized agents, including:
    - `PDFLoaderAgent`: Loads and extracts text from PDF files.
    - `PDFSummarizerAgent`: Summarizes the text of a single PDF using an LLM.
    - `ShortTermMemoryAgent`: Creates a temporary vector store from the summaries of the current session, enabling semantic search.
    - `DeepResearchSummarizerAgent`: Queries the short-term memory to synthesize a cross-document summary based on a user-provided query.
    - `LongTermMemoryAgent`: Manages a persistent, cumulative knowledge base by integrating summaries from the current session.
    - `ExperimentalDataLoaderAgent`: Loads and structures experimental data from a text file, if provided.
    - `KnowledgeIntegratorAgent`: Merges the deep research summary, long-term memory, and experimental data into a final, unified "knowledge brief."
    - `HypothesisGeneratorAgent`: Generates novel research hypotheses based on the integrated knowledge brief.
    - `ExperimentDesignerAgent`: Designs detailed experiments to test the generated hypotheses.
    - `ObserverAgent`: Reviews the outputs from all other agents to detect and report any errors.
  - **Output Handling:** Saves all generated artifacts (summaries, briefs, hypotheses, and experiment designs) into a structured project output directory.
- **Usage:** The orchestrator is invoked by the `gui.py` or can be run directly for command-line testing via `cli_test.py`.



### 3. `config.json` — Configuration File

- **Purpose:** Centralizes all settings, agent prompts, models, and workflow definitions.
- **Main Sections:**
  - **system_variables:** OpenAI API keys, model selection, output folder names, and timeouts.
  - **agent_prompts:** Customizable system prompts for each agent type.
  - **graph_definition:** Declarative description of the agent workflow (nodes and edges), specifying agent types, order, and data mappings.
- **Customization:** Users can adjust models, prompt styles, and workflow topology without modifying code.

<img width="572" height="810" alt="Untitled diagram - Copy _ Mermaid Chart-2025-08-26-174126" src="https://github.com/user-attachments/assets/b770ec25-c854-4aa6-8c70-282654474416" />

## Setup & Usage

### Installation

1. **Clone the repository:**
   ```
   git clone https://github.com/Juliolloret/MACS.git
   cd MACS
   ```

2. **Install dependencies:**
   ```
   pip install -r requirements.txt
   ```

   *Dependencies include PyQt6, openai, PyPDF2, reportlab, etc.*

3. **Configure OpenAI API Key:**
   - Add your API key to the `openai_api_key` field in `config.json`.

### Running the Application

#### GUI Mode

```bash
python gui.py
```
- Use the GUI to select your PDF folder, output directory, and optional experimental data/config.
- Click "Start Integrated Analysis" to launch the workflow.
- Monitor progress and status in the GUI.

#### Command-Line Interface (CLI)

Run MACS directly from the terminal using `cli.py`:

```bash
python cli.py --config config.json --pdf-dir path/to/pdfs --experimental-data path/to/data.txt --out-dir project_output
```

Add `--adaptive` to enable the adaptive orchestration cycle:

```bash
python cli.py --config config.json --pdf-dir path/to/pdfs --experimental-data path/to/data.txt --out-dir project_output --adaptive
```

For a self-contained test harness, the repository also includes `cli_test.py`.

1.  **Configure for CLI testing:** The script uses `config_cli_test_integrated.json`. If this file doesn't exist, the script will create a template for you. **You must edit this file to add a valid OpenAI API key.**
2.  **Run the test script:**
    ```bash
    python cli_test.py
    ```
    This will:
    - Create dummy PDF files and experimental data in the `cli_test_multi_input_pdfs/` and `cli_test_experimental_data/` directories.
    - Run the full orchestration pipeline using the settings from `config_cli_test_integrated.json`.
    - Save all outputs to a timestamped folder, e.g., `cli_test_integrated_project_output_YYYYMMDD_HHMMSS/`.

*Note: The agent workflow in the default `config_cli_test_integrated.json` may differ slightly from the primary `config.json` used by the GUI.*

---

## Workflow Details
MACS organizes research tasks into a graph of specialized agents defined in `config.json`. The standard workflow is as follows:

1. **PDF Loading:** The `PDFLoaderAgent` extracts text from all PDFs in the user-selected folder.
2. **Individual Summarization:** The `PDFSummarizerAgent` processes each PDF's text to create a concise summary.
3. **Short-Term Memory Creation:** The `ShortTermMemoryAgent` takes all the individual summaries and embeds them into a temporary vector store (using FAISS) for semantic search within the current session.
4. **Cross-Document Synthesis:** Based on a user query, the `DeepResearchSummarizerAgent` queries the short-term memory to find the most relevant summaries and synthesizes them into a coherent, cross-document answer.
5. **Long-Term Memory Integration:** The `LongTermMemoryAgent` integrates the new summaries into a persistent FAISS index, allowing knowledge to accumulate across different sessions.
6. **Experimental Data Integration:** (Optional) If a text file with experimental data is provided, the `ExperimentalDataLoaderAgent` loads and structures it.
7. **Knowledge Integration:** The `KnowledgeIntegratorAgent` is the central hub for synthesis. It merges the outputs from the `DeepResearchSummarizerAgent` (cross-document synthesis), the `LongTermMemoryAgent` (cumulative knowledge), and the `ExperimentalDataLoaderAgent` into a final, comprehensive "integrated knowledge brief."
8. **Hypothesis Generation:** The `HypothesisGeneratorAgent` uses this integrated brief to propose novel, testable research hypotheses.
9. **Experiment Design:** For each hypothesis, the `ExperimentDesignerAgent` outlines a detailed experimental plan.
10. **Observer Review:** Finally, the `ObserverAgent` scans the outputs of all previous agents to check for any errors or inconsistencies.
11. **Output:** All artifacts are saved in structured subfolders within the specified project output directory.


---

## Adaptive Evolution

MACS now includes an **adaptive evolution cycle** that can automatically refine its
agent graph. The runner in `adaptive/adaptive_graph_runner.py` executes the current
workflow, evaluates the results, and mutates the configuration for the next step.
This process repeats until a target quality threshold is achieved or a maximum
number of iterations is reached, allowing the system to improve its
performance iteratively.

---

## Run History and Reproducibility

Each orchestration run is assigned a unique `run_id`. Metadata about the run is
stored in `storage/run_history.jsonl`, and the full configuration is written to
`storage/run_configs/<run_id>.json`. Use the helpers in `storage.run_history`
to inspect past runs or to rerun a previous configuration:

```python
from storage.run_history import list_runs, get_run

runs = list_runs()
record = get_run(runs[-1]["run_id"])
config = record["config"]
prompt_ids = record.get("prompt_ids")
# pass `config` back into `run_project_orchestration` to reproduce the run
```

See [docs/run_history.md](docs/run_history.md) for more details.

---

## Configuration

**Modify `config.json` to:**
- Change LLM models per agent/task.
- Adjust agent prompts and instructions.
- Restructure the workflow graph (add, remove, or rewire agents).

---

## Extending & Contributing

We welcome contributions! Please read [Contributing Guide](CONTRIBUTING.md) for guidelines.
- Please note that this project is released with a [Code of Conduct](CODE_OF_CONDUCT.md). By participating, you agree to follow the terms.
- **Add New Agents:** Implement new agent classes in the `agents` package (files ending with `_agent.py`). They are auto-registered via the plugin system. Add them to the workflow graph in `config.json`.
- **Customize GUI:** Modify `gui.py` to add new options or workflow controls.
- **Pull Requests:** Contributions are welcome! Please open an issue or PR for discussion.

### Agent Plugins

Third-party agents can be added as plugins. Drop a Python module into the `agent_plugins/` directory and expose a `PLUGIN` variable:

```python
from agents.base_agent import Agent
from agents.registry import AgentPlugin, PluginMetadata, MACS_VERSION

class MyAgent(Agent):
    def run(self):
        ...

PLUGIN = AgentPlugin(
    agent_class=MyAgent,
    metadata=PluginMetadata(
        name="my_agent",
        version="0.1",
        author="ACME",
        macs_version=MACS_VERSION,
    ),
)
```

`load_plugins()` automatically discovers modules in this folder, registers compatible agents, and checks the `macs_version` field for compatibility.  
Additionally, plugins can be distributed as Python packages that expose a
`macs.plugins` entry point. After installing such a package with `pip`, the
plugin will be detected automatically.

To see all registered agents and plugins, run:

```bash
python cli.py --list-plugins
```

For a light-hearted example, check out `agent_plugins/skynet_agent.py`, which playfully references Skynet becoming self-aware.




---

## Security
If you find a security issue, please report it privately by email. Do not open public issues for security vulnerabilities.

---

## Credits

- Developed by Julio lloret Fillol. For academic, research, or non-commercial use.
- J-lloret-Fillol-LAB
- https://github.com/J-lloret-Fillol-LAB
- https://iciq.org/research-group/prof-julio-lloret-fillol/overview/
---

## Related Projects and Requirements
- Awesome Research Tools
- PyPDF2
- OpenAI
- PyQt6
- fastapi
- uvicorn
- python-dotenv
- langchain
- langchain-openai
- langchain-community
- tiktoken
- faiss-cpu
- lxml
- beautifulsoup4
- pandas
- requests
- reportlab


