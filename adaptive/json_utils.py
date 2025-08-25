from __future__ import annotations

import json
from typing import Any, Dict


def load_json(path: str) -> Dict[str, Any]:
    """Load JSON data from ``path``."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: str, data: Dict[str, Any]) -> None:
    """Persist ``data`` to ``path`` as JSON."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
