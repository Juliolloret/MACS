import os
import json
import uuid
from datetime import datetime
from typing import Any, Dict, Optional, List

from utils import log_status

RUN_HISTORY_PATH = os.path.join(os.path.dirname(__file__), "run_history.jsonl")
RUN_CONFIG_DIR = os.path.join(os.path.dirname(__file__), "run_configs")

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
    }

    os.makedirs(RUN_CONFIG_DIR, exist_ok=True)
    config_path = os.path.join(RUN_CONFIG_DIR, f"{run_id}.json")
    with open(config_path, "w", encoding="utf-8") as cf:
        json.dump(app_config, cf)
    record["config_path"] = os.path.relpath(config_path, os.path.dirname(RUN_HISTORY_PATH))

    if extra:
        record["extra"] = extra

    os.makedirs(os.path.dirname(RUN_HISTORY_PATH), exist_ok=True)
    with open(RUN_HISTORY_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
    log_status(f"[RunHistory] INFO: Recorded run {run_id}")

def get_run(run_id: str) -> Optional[Dict[str, Any]]:
    """Retrieve a run record and its full configuration by ``run_id``.

    Parameters
    ----------
    run_id: str
        Identifier returned by :func:`save_run`.

    Returns
    -------
    Optional[Dict[str, Any]]
        The stored record with the ``config`` key populated with the
        serialized application configuration, or ``None`` if the run is
        unknown.
    """
    if not os.path.exists(RUN_HISTORY_PATH):
        return None
    with open(RUN_HISTORY_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("run_id") == run_id:
                cfg_rel_path = rec.get("config_path")
                if cfg_rel_path:
                    cfg_abs_path = os.path.join(os.path.dirname(RUN_HISTORY_PATH), cfg_rel_path)
                    if os.path.exists(cfg_abs_path):
                        with open(cfg_abs_path, "r", encoding="utf-8") as cf:
                            rec["config"] = json.load(cf)
                    rec["config_path"] = cfg_abs_path
                return rec
    return None

def list_runs(include_config: bool = False) -> List[Dict[str, Any]]:
    """Return all stored run records.

    Parameters
    ----------
    include_config: bool, optional
        If ``True`` the full configuration for each run is loaded and
        attached to the record in a ``config`` key. Defaults to ``False``
        to keep listing lightweight.
    """
    if not os.path.exists(RUN_HISTORY_PATH):
        return []
    records: List[Dict[str, Any]] = []
    with open(RUN_HISTORY_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if include_config:
                cfg_rel_path = rec.get("config_path")
                if cfg_rel_path:
                    cfg_abs_path = os.path.join(os.path.dirname(RUN_HISTORY_PATH), cfg_rel_path)
                    if os.path.exists(cfg_abs_path):
                        with open(cfg_abs_path, "r", encoding="utf-8") as cf:
                            rec["config"] = json.load(cf)
                    rec["config_path"] = cfg_abs_path
            records.append(rec)
    return records
