# Run History and Reproducibility

Every call to `run_project_orchestration` gets a unique `run_id`. MACS writes a
small summary record to a JSON Lines file so you can track what was run without
saving full configs inline.

## Files Written

- `storage/run_history.jsonl`
  - One JSON object per line (JSON Lines format).
  - Each line contains the run_id, timestamp, config path, and input metadata.
- `storage/run_configs/<run_id>.json`
  - The full configuration that was actually used for that run.

## Record Schema

Each line in `run_history.jsonl` follows this structure:

```json
{
  "run_id": "<unique id>",
  "timestamp": "<UTC ISO timestamp>",
  "prompt_ids": ["..."],
  "config_path": "run_configs/<run_id>.json",
  "extra": {
    "pdf_file_paths": ["..."],
    "experimental_data_path": "...",
    "project_base_output_dir": "..."
  }
}
```

## How to Inspect Runs (Beginner Friendly)

### Option 1: Use the helper functions

```python
from storage.run_history import get_run, list_runs

runs = list_runs()
latest = get_run(runs[-1]["run_id"])
print(latest["run_id"])
print(latest["config"])
```

### Option 2: Open the JSONL file in a text editor

- Open `storage/run_history.jsonl`.
- Each line is a complete JSON object. Copy a line and paste it into an online
  JSON formatter if it is hard to read.

## Reproducing a Past Run

1. Load the config for the run.
2. Call `run_project_orchestration` again with the same inputs.

Example:

```python
from storage.run_history import get_run
from multi_agent_llm_system import run_project_orchestration

record = get_run("RUN_ID_HERE")
config = record["config"]
inputs = record["extra"]

run_project_orchestration(
    pdf_file_paths=inputs["pdf_file_paths"],
    experimental_data_path=inputs.get("experimental_data_path", ""),
    project_base_output_dir=inputs["project_base_output_dir"],
    app_config=config,
)
```

## Cleaning Old History

If the history file grows large, you can archive or delete old entries:

- Delete lines from `storage/run_history.jsonl` manually.
- Remove matching files in `storage/run_configs/` if you no longer need them.

Always keep backups if the runs are important for reproducibility.
