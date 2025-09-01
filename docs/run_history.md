# Run History and Reproducibility

Every call to `run_project_orchestration` is assigned a unique `run_id`. A
record of the run is appended to `storage/run_history.jsonl`, while the full
configuration used for that run is written to a separate JSON file. This
allows past runs to be inspected or replayed without turning the history log
into a large database of configs.

## Recording Runs

The orchestrator generates a `run_id` before executing the graph. The full
`app_config` is saved to `storage/run_configs/<run_id>.json`, and key runtime
parameters (PDF paths, experimental data path and output directory) are
serialized to a JSON Lines file. A record in `run_history.jsonl` has the
following schema:

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

## Inspecting or Restoring Runs

```python
from storage.run_history import get_run, list_runs

all_runs = list_runs()
last_run = get_run(all_runs[-1]["run_id"])
config = last_run["config"]
```

The retrieved `config` can be supplied directly to `run_project_orchestration`
to reproduce the workflow or to recover the `prompt_ids` that were active for
that run.
