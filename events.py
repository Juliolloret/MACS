"""Event bus utilities for orchestrator observability."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from threading import Lock
from typing import Any, Callable, Dict, List

EVENT_WORKFLOW_STARTED = "workflow_started"
EVENT_WORKFLOW_COMPLETED = "workflow_completed"
EVENT_NODE_STARTED = "node_started"
EVENT_NODE_COMPLETED = "node_completed"
EVENT_NODE_SKIPPED = "node_skipped"
EVENT_NODE_FAILED = "node_failed"
EVENT_NODE_RETRY = "node_retry"
EVENT_GRAPH_VISUALIZED = "graph_visualized"


@dataclass(frozen=True)
class Event:
    """Event payload wrapper passed to subscribers."""

    name: str
    payload: Dict[str, Any]


class EventBus:
    """Simple in-process event bus with optional wildcard subscribers."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._subscribers: Dict[str, List[Callable[[Event], None]]] = defaultdict(list)
        self._wildcard: List[Callable[[Event], None]] = []

    def subscribe(self, event_name: str, handler: Callable[[Event], None]) -> None:
        """Register a handler for an event name or ``*`` for all events."""
        if not callable(handler):
            raise ValueError("handler must be callable")
        with self._lock:
            if event_name == "*":
                self._wildcard.append(handler)
            else:
                self._subscribers[event_name].append(handler)

    def emit(self, event_name: str, payload: Dict[str, Any] | None = None) -> None:
        """Emit an event to subscribed handlers."""
        event = Event(event_name, payload or {})
        with self._lock:
            handlers = list(self._subscribers.get(event_name, []))
            wildcard = list(self._wildcard)
        for handler in handlers + wildcard:
            try:
                handler(event)
            except Exception:  # pragma: no cover - keep observability non-fatal
                continue


class NoopEventBus:
    """Event bus that discards all events."""

    def subscribe(self, _event_name: str, _handler: Callable[[Event], None]) -> None:
        return None

    def emit(self, _event_name: str, _payload: Dict[str, Any] | None = None) -> None:
        return None
