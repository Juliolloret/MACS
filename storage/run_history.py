import os
import json
import uuid
from datetime import datetime
from typing import Any, Dict, Optional, List

from utils import log_status

RUN_HISTORY_PATH = os.path.join(os.path.dirname(__file__), "run_history.jsonl")

def generate_run_id() -> str:
    """Return a new unique identifier for a run."""
    return uuid.uuid4().hex

def save_run(run_id: str, app_config: Dict[str, Any], extra: Optional[Dict[str, Any]] = None) -> None:
    """Persist a run record to the history log.

    Parameters
    ----------
    run_id: str
        Unique identifier for the run.
    app_config: Dict[str, Any]
        Full application configuration used for the run.
    extra: Optional[Dict[str, Any]]
        Additional runtime parameters such as file paths.
    """
    record = {
        "run_id": run_id,
        "timestamp": datetime.utcnow().isoformat(),
        # Explicitly capture prompt IDs if present
        "prompt_ids": app_config.get("system_variables", {}).get("prompt_ids"),
        "config": app_config,
    }
    if extra:
        record["extra"] = extra

    os.makedirs(os.path.dirname(RUN_HISTORY_PATH), exist_ok=True)
    with open(RUN_HISTORY_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
    log_status(f"[RunHistory] INFO: Recorded run {run_id}")

def get_run(run_id: str) -> Optional[Dict[str, Any]]:
    """Retrieve a run record by its identifier."""
    if not os.path.exists(RUN_HISTORY_PATH):
        return None
    with open(RUN_HISTORY_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("run_id") == run_id:
                return rec
    return None

def list_runs() -> List[Dict[str, Any]]:
    """Return all stored run records."""
    if not os.path.exists(RUN_HISTORY_PATH):
        return []
    records: List[Dict[str, Any]] = []
    with open(RUN_HISTORY_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records
