# MACS Development Guide

This guide explains how to set up, run, and extend MACS. It is written for
junior developers and assumes no prior experience with multi-agent systems.

## 1. What Is MACS?

MACS (Multi-Agent Collaboration System) runs a workflow of small, focused
agents. Each agent does one job (load PDFs, summarize text, create hypotheses).
The orchestrator connects them in a directed acyclic graph (DAG).

## 2. Requirements (First Time Setup)

### 2.1 Install Python

- Download Python from https://www.python.org
- Recommended: Python 3.10 or newer.
- Check your version:

```bash
python --version
```

### 2.2 Install Git (Optional but Recommended)

Git lets you clone and update the repository:

```bash
git --version
```

If you do not want Git, you can download the ZIP from GitHub and unzip it.

### 2.3 Create a Virtual Environment

A virtual environment keeps dependencies isolated.

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

### 2.4 Install Dependencies

```bash
python -m pip install -r requirements.txt
```

This installs GUI (PyQt), PDF parsing (pypdf), FAISS, and OpenAI SDK.

## 3. Dependency Auto-Check

MACS performs a dependency check when running the CLI or GUI:

- CLI: `python cli.py ...` always checks.
- GUI: `python gui.py` checks unless `--skip-dep-check` is used.
- GUI install helper: `python gui.py --install-deps` installs requirements
  before importing PyQt.

The check writes a cache file: `.cache/deps_state.json`. If the requirements
have not changed, MACS skips the check to run faster.

To force a refresh, set:

```bash
set MACS_DEPS_REFRESH=1
```

(or `export MACS_DEPS_REFRESH=1` on macOS/Linux).

## 4. Running MACS

### 4.1 GUI Mode

```bash
python gui.py
```

Steps inside the GUI:

1. Choose a PDF folder.
2. Choose an output folder.
3. (Optional) Choose an experimental data file.
4. Click "Start Integrated Analysis".

The GUI shows logs, graph images, and a button to open the latest metrics file.

### 4.2 CLI Mode

```bash
python cli.py --config config.json --pdf-dir path/to/pdfs --out-dir output_dir
```

Optional arguments:

- `--experimental-data path/to/data.txt`
- `--user-query "Your research question"`
- `--adaptive` for the adaptive cycle

## 5. Repository Structure (Beginner Map)

- `multi_agent_llm_system.py` - orchestrator and core execution.
- `agents/` - built-in agents.
- `agent_plugins/` - plugin agents.
- `adaptive/` - adaptive cycle and mutation tools.
- `docs/` - documentation.
- `tests/` - pytest-based tests.
- `config.json` - default workflow configuration.

## 6. Editing the Workflow

The workflow is defined in `config.json`. The key section is `graph_definition`.
If you add a node or change edges, validate the file:

```bash
python -c "from utils import load_app_config; load_app_config('config.json')"
```

For details, see `docs/CONFIGURATION_REFERENCE.md`.

## 7. Creating a New Agent (Step by Step)

1. Create a file in `agents/`, for example `agents/my_agent.py`.
2. Subclass `Agent` and implement `execute(self, inputs)`.
3. Register the class with `@register_agent("MyAgent")`.
4. Add a node with type `MyAgent` in `config.json`.
5. Write a unit test in `tests/`.

Skeleton:

```python
from agents.base_agent import Agent
from agents.registry import register_agent

@register_agent("MyAgent")
class MyAgent(Agent):
    def execute(self, inputs: dict) -> dict:
        text = inputs.get("text", "")
        return {"output": text.upper()}
```

## 8. Testing

Run all tests:

```bash
python scripts/run_tests.py
```

This script runs pytest and writes the output to `test_results.txt` for review.

If you only want pytest:

```bash
pytest
```

## 9. Debugging Tips

- Check `node_metrics.json` in the output folder.
- Look at GUI logs or CLI console output.
- If an agent returns an `error` key, downstream nodes may be skipped.
- Use small PDF files while debugging to speed up runs.

## 10. Documentation Practices

- Update README and docs when behavior changes.
- Keep examples copy-paste friendly.
- Avoid advanced jargon without explanation.

## 11. Getting Help

- Review `README.md` and docs in `docs/`.
- Search for existing issues in the repository.
- If stuck, open a GitHub issue with error logs and steps to reproduce.
