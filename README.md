Here’s a detailed extension for your README.md, incorporating the structure and function of gui.py, multi_agent_llm_system.py, and config.json:

---

# MACS: Multi-Agent LLM Research Assistant

## Overview

**MACS** (Multi-Agent Collaboration System) is a modular Python-based system for orchestrating advanced research workflows on collections of academic PDFs. It integrates multiple specialized agents—powered by large language models (LLMs)—to automate PDF loading, summarization, multi-document synthesis, web research, experimental data integration, hypothesis generation, and experiment design. The system features both a GUI (built with PyQt6) and a flexible configuration system.

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

---

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

- **Add New Agents:** Implement new agent classes in `multi_agent_llm_system.py` and add them to the graph in `config.json`.
- **Customize GUI:** Edit `gui.py` to add new options or workflow controls.
- **Pull Requests:** Contributions are welcome! Please open an issue or PR for discussion.

---

## License

MIT License
Copyright (c) 2025 Juliolloret

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

---

## Credits

Developed by Juliolloret. For academic, research, or non-commercial use.

---

Let me know if you want this as a Markdown file or if you want sections broken down differently!
