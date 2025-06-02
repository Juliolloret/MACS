<p align="center">
  <img src="https://github.com/user-attachments/assets/e2d9cd31-4e54-41f9-81d6-11304b0f80c1" alt="MACS Logo" width="250"/>
</p>

![GitHub Workflow Status](https://img.shields.io/github/workflow/status/Juliolloret/MACS/CI)
![License](https://img.shields.io/github/license/Juliolloret/MACS)
![Python Version](https://img.shields.io/badge/python-3.8+-blue)

---

# MACS: Multi-Agent LLM Research Assistant
**MACS** (Multi-Agent Collaboration System) is a modular Python-based system for orchestrating advanced research workflows on collections of academic PDFs. 
 It integrates multiple specialized agents to automate literature review, synthesis, and experimental planning. 

## Table of Contents

- [Overview](#overview)
- [Components](#components)
- [Setup & Usage](#setup--usage)
- [Workflow Details](#workflow-details)
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

### 2. `multi_agent_llm_system.py` — Multi-Agent Orchestration Backend

- **Purpose:** Core backend that defines, manages, and executes the multi-agent workflow.
- **Key Elements:**
  - **Config Loader:** Reads `config.json` for system settings, agent prompts, and the agent graph structure.
  - **Agents:** Each node in the workflow is an agent class specializing in a research task:
    - `PDFLoaderAgent`: Loads and extracts text from PDFs.
    - `PDFSummarizerAgent`: Summarizes single PDFs with LLMs.
    - `MultiDocSynthesizerAgent`: Synthesizes knowledge across multiple summaries.
    - `WebResearcherAgent`: Simulates web research based on synthesized understanding.
    - `ExperimentalDataLoaderAgent`: Loads and summarizes experimental data (if provided).
    - `KnowledgeIntegratorAgent`: Integrates all knowledge into a unified brief.
    - `HypothesisGeneratorAgent`: Generates research hypotheses.
    - `ExperimentDesignerAgent`: Designs experiments for generated hypotheses.
  - **Graph Orchestrator:** Executes the workflow graph as defined in `config.json`, ensuring correct data flow and error handling.
  - **Output Handling:** Saves synthesized outputs, hypotheses, and experiment designs into organized subfolders.
- **Usage:** Can be called from the GUI or invoked directly for CLI testing.



### 3. `config.json` — Configuration File

- **Purpose:** Centralizes all settings, agent prompts, models, and workflow definitions.
- **Main Sections:**
  - **system_variables:** OpenAI API keys, model selection, output folder names, and timeouts.
  - **agent_prompts:** Customizable system prompts for each agent type.
  - **graph_definition:** Declarative description of the agent workflow (nodes and edges), specifying agent types, order, and data mappings.
- **Customization:** Users can adjust models, prompt styles, and workflow topology without modifying code.

<img src="https://github.com/user-attachments/assets/34fc348b-cbfb-48ca-a187-b5b2b94f7f9b" width="350" height="680"/>


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

#### Backend/CLI Mode

For advanced/automated workflows, invoke the orchestrator directly:
```bash
python multi_agent_llm_system.py
```
(See code comments in `multi_agent_llm_system.py` for CLI test usage.)

---

## Workflow Details
The current JSON file implements the following workflow. However, you can modify it by changing the JSON file. 
1. **PDF Loading:** Extracts text from all PDFs in the selected folder.
2. **Summarization:** Each PDF is summarized by an LLM agent.
3. **Multi-Document Synthesis:** Summaries are synthesized into a cross-document understanding.
4. **Web Research (LLM-based):** (Optional) Adds simulated web research.
5. **Experimental Data Integration:** (Optional) Loads and summarizes experimental results.
6. **Knowledge Integration:** Combines all sources into an integrated knowledge brief.
7. **Hypothesis Generation:** Proposes new hypotheses based on the brief.
8. **Experiment Design:** Designs experiments for each hypothesis.
9. **Output:** All results are saved in structured subfolders in the project output directory.

---

## Configuration

**Modify `config.json` to:**
- Change LLM models per agent/task.
- Adjust agent prompts and instructions.
- Restructure the workflow graph (add, remove, or rewire agents).

---

## Extending & Contributing

We welcome contributions! Please read CONTRIBUTING.md for guidelines.
Please note that this project is released with a [Code of Conduct](CODE_OF_CONDUCT.md). By participating, you agree to follow the terms.
- **Add New Agents:** Implement new agent classes in `multi_agent_llm_system.py` and add them to the graph in `config.json`.
- **Customize GUI:** Edit `gui.py` to add new options or workflow controls.
- **Pull Requests:** Contributions are welcome! Please open an issue or PR for discussion.


---

## Security
If you find a security issue, please report it privately by email to juliolloret@example.com (replace with your contact). Do not open public issues for security vulnerabilities.

---

## Credits

- Developed by Julio lloret Fillol. For academic, research, or non-commercial use.
- J-lloret-Fillol-LAB
- https://github.com/J-lloret-Fillol-LAB
- https://iciq.org/research-group/prof-julio-lloret-fillol/overview/
---

## Related Projects
- Awesome Research Tools
- PyPDF2
- OpenAI
- PyQt6
---

