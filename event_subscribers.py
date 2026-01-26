"""Event bus subscribers for logging and metrics collection."""

from __future__ import annotations

import json
from collections import defaultdict
from typing import Callable, Dict, List

from events import (
    Event,
    EVENT_NODE_COMPLETED,
    EVENT_NODE_FAILED,
    EVENT_NODE_RETRY,
    EVENT_NODE_SKIPPED,
    EVENT_NODE_STARTED,
    EVENT_WORKFLOW_COMPLETED,
    EVENT_WORKFLOW_STARTED,
)
from utils import log_status


class EventLogger:
    """Subscriber that logs event payloads using ``log_status``."""

    def __init__(self, logger: Callable[[str], None] | None = None) -> None:
        self._logger = logger or log_status

    def handle(self, event: Event) -> None:
        """Log the event name and payload."""
        self._logger(f"[EventBus] {event.name}: {event.payload}")


class MetricsCollector:
    """Subscriber that aggregates node metrics, retries, skips, and failures.

    This enhanced collector tracks:
    - Node execution metrics (time, tokens)
    - Retry attempts per node
    - Skipped nodes and reasons
    - Failed nodes and errors
    - Event counts by type
    """

    def __init__(self, output_path: str | None = None) -> None:
        self.event_counts: Dict[str, int] = defaultdict(int)
        self.node_metrics: Dict[str, Dict[str, object]] = {}
        self.retries: Dict[str, List[Dict[str, object]]] = defaultdict(list)
        self.skipped: Dict[str, str] = {}
        self.failures: Dict[str, str] = {}
        self.workflow: Dict[str, Dict[str, object] | None] = {
            "started": None,
            "completed": None,
        }
        self._output_path = output_path
        self._written = False

    def handle(self, event: Event) -> None:
        """Capture metrics from the event stream."""
        self.event_counts[event.name] += 1

        if event.name == EVENT_NODE_STARTED:
            node_id = event.payload.get("node_id")
            if node_id and node_id not in self.node_metrics:
                self.node_metrics[node_id] = {"started": True}

        elif event.name == EVENT_NODE_COMPLETED:
            node_id = event.payload.get("node_id")
            if node_id:
                self.node_metrics[node_id] = {
                    "elapsed_sec": event.payload.get("elapsed_sec"),
                    "tokens_used": event.payload.get("tokens_used"),
                    "error": event.payload.get("error"),
                }

        elif event.name == EVENT_NODE_RETRY:
            node_id = event.payload.get("node_id")
            if node_id:
                self.retries[node_id].append({
                    "attempt": event.payload.get("attempt"),
                    "error": event.payload.get("error"),
                })

        elif event.name == EVENT_NODE_SKIPPED:
            node_id = event.payload.get("node_id")
            if node_id:
                self.skipped[node_id] = event.payload.get("reason", "Unknown")

        elif event.name == EVENT_NODE_FAILED:
            node_id = event.payload.get("node_id")
            if node_id:
                self.failures[node_id] = event.payload.get("error", "Unknown error")

        elif event.name == EVENT_WORKFLOW_STARTED:
            self.workflow["started"] = dict(event.payload)

        elif event.name == EVENT_WORKFLOW_COMPLETED:
            self.workflow["completed"] = dict(event.payload)
            if self._output_path and not self._written:
                self.write_json(self._output_path)
                self._written = True

    def to_dict(self) -> Dict[str, object]:
        """Return a JSON-serializable snapshot of metrics."""
        return {
            "event_counts": dict(self.event_counts),
            "node_metrics": dict(self.node_metrics),
            "retries": dict(self.retries),
            "skipped": dict(self.skipped),
            "failures": dict(self.failures),
            "workflow": dict(self.workflow),
        }

    def get_summary(self) -> Dict[str, object]:
        """Return a summary of the workflow execution."""
        return {
            "total_nodes": len(self.node_metrics),
            "successful_nodes": len([n for n, m in self.node_metrics.items()
                                     if not m.get("error") and n not in self.skipped]),
            "failed_nodes": len(self.failures),
            "skipped_nodes": len(self.skipped),
            "retry_count": sum(len(r) for r in self.retries.values()),
        }

    def write_json(self, path: str) -> None:
        """Persist the collected metrics to disk."""
        data = self.to_dict()
        data["summary"] = self.get_summary()
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2)

