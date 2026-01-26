<p align="center">
  <img src="https://github.com/user-attachments/assets/e2d9cd31-4e54-41f9-81d6-11304b0f80c1" alt="MACS Logo" width="250"/>
</p>

![GitHub Workflow Status](https://img.shields.io/github/workflow/status/Juliolloret/MACS/CI)
![License](https://img.shields.io/github/license/Juliolloret/MACS)
![Python Version](https://img.shields.io/badge/python-3.8+-blue)

---

# MACS: Multi-Agent LLM Research Assistant

MACS (Multi-Agent Collaboration System) is a modular Python platform that runs
research workflows on collections of PDFs. It uses a graph of small agents that
load documents, summarize them, synthesize findings, and propose hypotheses.

## Who This Is For

- Students or researchers who want automated literature review support.
- Developers who want a configurable multi-agent pipeline.
- Beginners who want to learn how agent graphs work in practice.

## Quick Start (Beginner Friendly)

### 0. Prerequisites

- Python 3.10+ recommended (3.8+ may work depending on dependencies).
- Git (optional, but recommended).

Check your Python version:

```bash
python --version
```

### 1. Get the Code

```bash
git clone https://github.com/Juliolloret/MACS.git
cd MACS
```

### 2. Create and Activate a Virtual Environment

Windows (PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Configure Your API Key

Open `config.json` and set `system_variables.openai_api_key`. Alternatively set
`OPENAI_API_KEY` in your environment.

### 5. Run the GUI

```bash
python gui.py
```

If PyQt is missing, you can install dependencies first:

```bash
python gui.py --install-deps
```

Note: `--install-deps` respects `predefined_routes.ini` and only installs when
`auto_install_deps = true`.

### 6. Run the CLI

```bash
python cli.py --config config.json --pdf-dir path/to/pdfs --out-dir output_dir
```

Optional flags:

- `--experimental-data path/to/data.txt`
- `--user-query "Your research question"`
- `--adaptive` to run the adaptive cycle

### 7. Check the Output

Outputs are written under your output directory. See the "Output Structure"
section below.

## Documentation Index

- `README.md` - project overview and quick start
- `docs/DEVELOPMENT_GUIDE.md` - detailed developer setup and workflow
- `docs/CONFIGURATION_REFERENCE.md` - config.json explained
- `docs/CODE_SCHEME.md` - high-level architecture map
- `docs/RESPONSES_API_MIGRATION.md` - OpenAI Responses API notes
- `docs/run_history.md` - run history and reproducibility
- `AGENTS.md` - agent reference and extension guide
- `memory.md` - memory system details

## How MACS Works (Simple Explanation)

1. The orchestrator reads `config.json` and builds a DAG of agents.
2. Agents run in order, passing outputs to the next agent.
3. A metrics file (`node_metrics.json`) is generated when the run completes.
4. Output files (summaries, briefs, hypotheses) are saved to your output folder.

## Features

- Automated literature review from PDFs.
- Multi-document synthesis and knowledge integration.
- Hypothesis generation and experiment design.
- Optional adaptive graph evolution.
- GUI and CLI entrypoints.
- Event-driven metrics and logs for observability.

## Graph Visualization

MACS generates interactive HTML graphs during workflow execution:

| File | When Generated |
|------|----------------|
| `graph_initial.html` | Before execution starts |
| `graph_{node}.html` | During each node execution (current node highlighted) |
| `graph_final.html` | After workflow completes |

**Features:**
- Hierarchical top-down layout showing DAG flow
- Click any node to view agent details in an info panel
- Color-coded states: blue (pending), orange (executing), green (completed), red (failed)
- Hover tooltips with agent type and execution metrics

## Output Structure

Typical output directory:

```
project_output/
  node_metrics.json
  pdf_summaries/
  project_synthesis/
    multi_document_synthesis.txt
    web_research_summary.txt
    integrated_knowledge_brief.txt
    experimental_data_summary.txt
  project_hypotheses/
    hypotheses_raw_llm_output.json
    hypotheses_list.txt
    key_research_opportunities.txt
  adaptive_metrics/            (only for adaptive runs)
```

## Adaptive Evolution

Adaptive mode runs a loop of run -> evaluate -> mutate. Metrics for each cycle
are stored in `adaptive_metrics/node_metrics_cycle_<n>.json`.

Run adaptive mode from the CLI:

```bash
python cli.py --config config.json --pdf-dir path/to/pdfs --out-dir output_dir --adaptive
```

## Testing

Run all tests and save output to `test_results.txt`:

```bash
python scripts/run_tests.py
```

You can also run pytest directly:

```bash
pytest
```

## Troubleshooting (Common Issues)

- "No module named PyQt" when launching GUI
  - Run `python gui.py --install-deps` or `pip install -r requirements.txt`.
- "openai_api_key missing"
  - Set `OPENAI_API_KEY` or edit `config.json`.
- PDF loading errors
  - Ensure the files are not encrypted and are valid PDFs.
- FAISS errors
  - Install `faiss-cpu` and `langchain-community`.

## Extending MACS

To add a new agent:

1. Create a class in `agents/` or `agent_plugins/`.
2. Register it with `@register_agent("YourAgent")` or plugin metadata.
3. Add it to `config.json` and wire it with edges.
4. Add tests in `tests/`.

See `AGENTS.md` for a full guide.

## Security

If you find a security issue, please report it privately by email. Do not open
public issues for security vulnerabilities.

## Credits

- Developed by Julio lloret Fillol.
- J-lloret-Fillol-LAB
- https://github.com/J-lloret-Fillol-LAB
- https://iciq.org/research-group/prof-julio-lloret-fillol/overview/

## Related Projects and Requirements

- OpenAI
- PyQt6
- pypdf
- langchain
- langchain-openai
- langchain-community
- tiktoken
- faiss-cpu
- pandas
- requests
- reportlab
- pyvis
