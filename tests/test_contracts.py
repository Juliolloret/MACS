"""Tests for contracts module."""

import pytest

from contracts import (
    NodeContext,
    NodeResult,
    AgentCapabilities,
    UpstreamIssue,
    InputGatherResult,
)


class TestNodeContext:
    """Tests for NodeContext dataclass."""

    def test_create_context(self):
        """Test creating a node context."""
        ctx = NodeContext(
            run_id="run-123",
            node_id="test_node",
            attempt=1,
            project_base_output_dir="/output",
        )

        assert ctx.run_id == "run-123"
        assert ctx.node_id == "test_node"
        assert ctx.attempt == 1
        assert ctx.project_base_output_dir == "/output"
        assert ctx.requires_project_dir is False
        assert ctx.consumes_all_outputs is False

    def test_context_with_capabilities(self):
        """Test creating context with capability flags."""
        ctx = NodeContext(
            run_id="run-123",
            node_id="memory_node",
            attempt=1,
            project_base_output_dir="/output",
            requires_project_dir=True,
        )

        assert ctx.requires_project_dir is True


class TestNodeResult:
    """Tests for NodeResult dataclass."""

    def test_create_empty_result(self):
        """Test creating an empty result."""
        result = NodeResult()

        assert result.data == {}
        assert result.errors == []
        assert result.warnings == []
        assert result.has_error is False
        assert result.error is None

    def test_result_with_data(self):
        """Test creating a result with data."""
        result = NodeResult(data={"output": "value"})

        assert result.data["output"] == "value"
        assert result.has_error is False

    def test_result_with_error(self):
        """Test creating a result with errors."""
        result = NodeResult(errors=["Error 1", "Error 2"])

        assert result.has_error is True
        assert result.error == "Error 1; Error 2"

    def test_to_dict(self):
        """Test converting result to dictionary."""
        result = NodeResult(
            data={"key": "value"},
            errors=["something went wrong"],
            warnings=["minor issue"],
        )

        d = result.to_dict()

        assert d["key"] == "value"
        assert d["error"] == "something went wrong"
        assert d["warnings"] == ["minor issue"]

    def test_to_dict_no_error(self):
        """Test that to_dict excludes error key when no errors."""
        result = NodeResult(data={"key": "value"})

        d = result.to_dict()

        assert "error" not in d
        assert d["key"] == "value"

    def test_from_dict(self):
        """Test creating result from legacy dictionary."""
        legacy = {
            "output": "data",
            "error": "something failed",
            "warnings": ["warning1"],
        }

        result = NodeResult.from_dict(legacy)

        assert result.data["output"] == "data"
        assert result.errors == ["something failed"]
        assert result.warnings == ["warning1"]

    def test_from_dict_no_error(self):
        """Test creating result from dict without error."""
        legacy = {"output": "data"}

        result = NodeResult.from_dict(legacy)

        assert result.data["output"] == "data"
        assert result.has_error is False


class TestAgentCapabilities:
    """Tests for AgentCapabilities dataclass."""

    def test_default_capabilities(self):
        """Test default capability values."""
        caps = AgentCapabilities.default()

        assert caps.requires_project_dir is False
        assert caps.consumes_all_outputs is False
        assert caps.needs_input_keys == []
        assert caps.optional_input_keys == []
        assert caps.loop_over is None
        assert caps.parallelizable is False

    def test_custom_capabilities(self):
        """Test creating custom capabilities."""
        caps = AgentCapabilities(
            requires_project_dir=True,
            consumes_all_outputs=True,
            needs_input_keys=["input1"],
            loop_over="items",
            parallelizable=True,
        )

        assert caps.requires_project_dir is True
        assert caps.consumes_all_outputs is True
        assert caps.needs_input_keys == ["input1"]
        assert caps.loop_over == "items"
        assert caps.parallelizable is True


class TestUpstreamIssue:
    """Tests for UpstreamIssue dataclass."""

    def test_create_issue(self):
        """Test creating an upstream issue."""
        issue = UpstreamIssue(
            source_node="source",
            target_key="input_key",
            message="Missing data",
            fatal=True,
        )

        assert issue.source_node == "source"
        assert issue.target_key == "input_key"
        assert issue.message == "Missing data"
        assert issue.fatal is True

    def test_non_fatal_by_default(self):
        """Test that issues are non-fatal by default."""
        issue = UpstreamIssue(
            source_node="source",
            target_key="key",
            message="Warning",
        )

        assert issue.fatal is False


class TestInputGatherResult:
    """Tests for InputGatherResult dataclass."""

    def test_empty_result(self):
        """Test creating an empty gather result."""
        result = InputGatherResult()

        assert result.inputs == {}
        assert result.issues == []
        assert result.has_fatal_error is False
        assert result.error_messages == []

    def test_with_fatal_issue(self):
        """Test gather result with fatal issue."""
        fatal_issue = UpstreamIssue(
            source_node="src",
            target_key="key",
            message="Fatal error",
            fatal=True,
        )
        result = InputGatherResult(issues=[fatal_issue])

        assert result.has_fatal_error is True
        assert result.error_messages == ["Fatal error"]

    def test_with_non_fatal_issues(self):
        """Test gather result with non-fatal issues."""
        issue = UpstreamIssue(
            source_node="src",
            target_key="key",
            message="Warning only",
            fatal=False,
        )
        result = InputGatherResult(issues=[issue])

        assert result.has_fatal_error is False
        assert result.error_messages == ["Warning only"]
