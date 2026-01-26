"""Structured logging utilities with context support.

This module provides a lightweight structured logger that replaces the
simple `log_status` pattern with context-aware logging.
"""

from __future__ import annotations

import json
import logging
import sys
from contextvars import ContextVar
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, Optional

# Context variables for automatic inclusion in log records
_run_id: ContextVar[Optional[str]] = ContextVar("run_id", default=None)
_node_id: ContextVar[Optional[str]] = ContextVar("node_id", default=None)
_attempt: ContextVar[int] = ContextVar("attempt", default=1)


@dataclass
class LogContext:
    """Context information automatically included in log records.

    Attributes
    ----------
    run_id : Optional[str]
        Current workflow run identifier.
    node_id : Optional[str]
        Current node being executed.
    attempt : int
        Current execution attempt (1-indexed).
    timestamp : str
        ISO format timestamp.
    """

    run_id: Optional[str] = None
    node_id: Optional[str] = None
    attempt: int = 1
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    @classmethod
    def current(cls) -> "LogContext":
        """Create a LogContext from current context variables."""
        return cls(
            run_id=_run_id.get(),
            node_id=_node_id.get(),
            attempt=_attempt.get(),
        )


class ContextManager:
    """Manage context variables for structured logging.

    Usage::

        with ContextManager(run_id="abc123", node_id="summarizer"):
            logger.info("Processing node")  # includes run_id and node_id
    """

    def __init__(
        self,
        run_id: Optional[str] = None,
        node_id: Optional[str] = None,
        attempt: Optional[int] = None,
    ):
        self.run_id = run_id
        self.node_id = node_id
        self.attempt = attempt
        self._tokens: Dict[str, Any] = {}

    def __enter__(self) -> "ContextManager":
        if self.run_id is not None:
            self._tokens["run_id"] = _run_id.set(self.run_id)
        if self.node_id is not None:
            self._tokens["node_id"] = _node_id.set(self.node_id)
        if self.attempt is not None:
            self._tokens["attempt"] = _attempt.set(self.attempt)
        return self

    def __exit__(self, *args: Any) -> None:
        if "run_id" in self._tokens:
            _run_id.reset(self._tokens["run_id"])
        if "node_id" in self._tokens:
            _node_id.reset(self._tokens["node_id"])
        if "attempt" in self._tokens:
            _attempt.reset(self._tokens["attempt"])


def set_context(
    run_id: Optional[str] = None,
    node_id: Optional[str] = None,
    attempt: Optional[int] = None,
) -> None:
    """Set context variables for structured logging.

    Parameters
    ----------
    run_id : Optional[str]
        Workflow run identifier.
    node_id : Optional[str]
        Current node identifier.
    attempt : Optional[int]
        Current attempt number.
    """
    if run_id is not None:
        _run_id.set(run_id)
    if node_id is not None:
        _node_id.set(node_id)
    if attempt is not None:
        _attempt.set(attempt)


def get_context() -> LogContext:
    """Get the current logging context."""
    return LogContext.current()


class StructuredLogger:
    """Logger that includes context in all messages.

    This is a lightweight wrapper that can be used alongside the existing
    `log_status` function while providing structured context.

    Parameters
    ----------
    name : str
        Logger name (typically module name).
    callback : Optional[Callable[[str], None]]
        Optional callback for log messages (for GUI integration).
    """

    def __init__(
        self,
        name: str,
        callback: Optional[Callable[[str], None]] = None,
    ):
        self.name = name
        self.callback = callback
        self._logger = logging.getLogger(name)

    def _format_message(
        self,
        level: str,
        message: str,
        extra: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Format a log message with context."""
        ctx = get_context()
        parts = []

        # Add context prefix
        if ctx.run_id:
            parts.append(f"[run:{ctx.run_id[:8]}]")
        if ctx.node_id:
            parts.append(f"[{ctx.node_id}]")
            if ctx.attempt > 1:
                parts.append(f"[attempt:{ctx.attempt}]")

        parts.append(message)

        return " ".join(parts)

    def _log(
        self,
        level: str,
        message: str,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Internal log method."""
        formatted = self._format_message(level, message, extra)

        # Log to Python logger
        log_level = getattr(logging, level.upper(), logging.INFO)
        self._logger.log(log_level, formatted)

        # Also call callback if set (for GUI)
        if self.callback:
            self.callback(formatted)

    def debug(self, message: str, **kwargs: Any) -> None:
        """Log a debug message."""
        self._log("debug", message, kwargs or None)

    def info(self, message: str, **kwargs: Any) -> None:
        """Log an info message."""
        self._log("info", message, kwargs or None)

    def warning(self, message: str, **kwargs: Any) -> None:
        """Log a warning message."""
        self._log("warning", message, kwargs or None)

    def error(self, message: str, **kwargs: Any) -> None:
        """Log an error message."""
        self._log("error", message, kwargs or None)

    def critical(self, message: str, **kwargs: Any) -> None:
        """Log a critical message."""
        self._log("critical", message, kwargs or None)


# Global logger instance for convenience
_default_logger: Optional[StructuredLogger] = None


def get_logger(name: str = "macs") -> StructuredLogger:
    """Get or create a structured logger.

    Parameters
    ----------
    name : str
        Logger name.

    Returns
    -------
    StructuredLogger
        The logger instance.
    """
    global _default_logger
    if _default_logger is None or _default_logger.name != name:
        _default_logger = StructuredLogger(name)
    return _default_logger


def configure_logging(
    level: int = logging.INFO,
    callback: Optional[Callable[[str], None]] = None,
) -> None:
    """Configure the logging system.

    Parameters
    ----------
    level : int
        Logging level (e.g., logging.INFO).
    callback : Optional[Callable[[str], None]]
        Optional callback for all log messages.
    """
    global _default_logger
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        stream=sys.stdout,
    )
    if callback:
        _default_logger = StructuredLogger("macs", callback=callback)


# Export for integration with existing log_status
def structured_log_status(message: str) -> None:
    """Replacement for log_status that includes context.

    This can be used as a drop-in replacement for log_status
    that automatically includes run_id and node_id context.
    """
    logger = get_logger()
    logger.info(message)
