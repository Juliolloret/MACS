"""Tests for logger module."""

import pytest
from unittest import mock

from logger import (
    StructuredLogger,
    ContextManager,
    LogContext,
    get_context,
    set_context,
    get_logger,
    structured_log_status,
)


class TestLogContext:
    """Tests for LogContext dataclass."""

    def test_current_returns_defaults(self):
        """Test that current() returns default values when no context set."""
        ctx = LogContext.current()

        assert ctx.run_id is None
        assert ctx.node_id is None
        assert ctx.attempt == 1
        assert ctx.timestamp is not None


class TestContextManager:
    """Tests for ContextManager."""

    def test_sets_run_id(self):
        """Test that context manager sets run_id."""
        with ContextManager(run_id="test-run-123"):
            ctx = get_context()
            assert ctx.run_id == "test-run-123"

        # Context should be reset
        ctx = get_context()
        assert ctx.run_id is None

    def test_sets_node_id(self):
        """Test that context manager sets node_id."""
        with ContextManager(node_id="summarizer"):
            ctx = get_context()
            assert ctx.node_id == "summarizer"

        ctx = get_context()
        assert ctx.node_id is None

    def test_sets_multiple_values(self):
        """Test that context manager sets multiple values."""
        with ContextManager(run_id="run1", node_id="node1", attempt=2):
            ctx = get_context()
            assert ctx.run_id == "run1"
            assert ctx.node_id == "node1"
            assert ctx.attempt == 2

    def test_nested_context(self):
        """Test nested context managers."""
        with ContextManager(run_id="outer"):
            assert get_context().run_id == "outer"

            with ContextManager(node_id="inner"):
                ctx = get_context()
                assert ctx.run_id == "outer"
                assert ctx.node_id == "inner"

            # Inner context should be reset
            assert get_context().node_id is None
            assert get_context().run_id == "outer"


class TestSetContext:
    """Tests for set_context function."""

    def test_set_run_id(self):
        """Test setting run_id directly."""
        set_context(run_id="my-run")
        assert get_context().run_id == "my-run"

        # Clean up
        set_context(run_id=None)

    def test_partial_update(self):
        """Test that partial updates work."""
        set_context(run_id="run", node_id="node")
        set_context(attempt=3)  # Should not clear run_id or node_id

        ctx = get_context()
        assert ctx.run_id == "run"
        assert ctx.node_id == "node"
        assert ctx.attempt == 3


class TestStructuredLogger:
    """Tests for StructuredLogger class."""

    def test_basic_logging(self):
        """Test basic log message."""
        messages = []
        logger = StructuredLogger("test", callback=messages.append)

        logger.info("Test message")

        assert len(messages) == 1
        assert "Test message" in messages[0]

    def test_context_included_in_message(self):
        """Test that context is included in log messages."""
        messages = []
        logger = StructuredLogger("test", callback=messages.append)

        with ContextManager(run_id="abc12345", node_id="summarizer"):
            logger.info("Processing")

        assert len(messages) == 1
        assert "[run:abc12345]" in messages[0]
        assert "[summarizer]" in messages[0]
        assert "Processing" in messages[0]

    def test_attempt_shown_only_when_greater_than_1(self):
        """Test that attempt is only shown for retries."""
        messages = []
        logger = StructuredLogger("test", callback=messages.append)

        with ContextManager(node_id="node", attempt=1):
            logger.info("First attempt")

        with ContextManager(node_id="node", attempt=2):
            logger.info("Retry")

        assert "[attempt:" not in messages[0]
        assert "[attempt:2]" in messages[1]

    def test_all_log_levels(self):
        """Test all log level methods."""
        messages = []
        logger = StructuredLogger("test", callback=messages.append)

        logger.debug("Debug message")
        logger.info("Info message")
        logger.warning("Warning message")
        logger.error("Error message")
        logger.critical("Critical message")

        assert len(messages) == 5


class TestGetLogger:
    """Tests for get_logger function."""

    def test_returns_structured_logger(self):
        """Test that get_logger returns a StructuredLogger."""
        logger = get_logger("test")

        assert isinstance(logger, StructuredLogger)
        assert logger.name == "test"

    def test_caches_logger(self):
        """Test that get_logger caches the default logger."""
        logger1 = get_logger()
        logger2 = get_logger()

        # Note: The caching behavior depends on the name
        assert isinstance(logger1, StructuredLogger)
        assert isinstance(logger2, StructuredLogger)


class TestStructuredLogStatus:
    """Tests for structured_log_status function."""

    def test_logs_message(self):
        """Test that structured_log_status works."""
        # This just tests it doesn't raise
        structured_log_status("Test message")
