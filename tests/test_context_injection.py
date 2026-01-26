"""Tests for context_injection module."""

import pytest

from context_injection import (
    get_agent_capabilities,
    gather_inputs,
    inject_context,
    append_upstream_issue,
    AGENTS_REQUIRING_PROJECT_DIR,
    AGENTS_CONSUMING_ALL_OUTPUTS,
)
from contracts import AgentCapabilities, UpstreamIssue


class TestGetAgentCapabilities:
    """Tests for get_agent_capabilities function."""

    def test_knowledge_integrator_requires_project_dir(self):
        """Test that KnowledgeIntegratorAgent requires project dir."""
        caps = get_agent_capabilities("KnowledgeIntegratorAgent", {})

        assert caps.requires_project_dir is True

    def test_observer_consumes_all_outputs(self):
        """Test that ObserverAgent consumes all outputs."""
        caps = get_agent_capabilities("ObserverAgent", {})

        assert caps.consumes_all_outputs is True

    def test_basic_agent_has_no_special_requirements(self):
        """Test that basic agents have no special requirements."""
        caps = get_agent_capabilities("PDFSummarizerAgent", {})

        assert caps.requires_project_dir is False
        assert caps.consumes_all_outputs is False

    def test_loop_config_is_extracted(self):
        """Test that loop configuration is extracted from config."""
        caps = get_agent_capabilities("TestAgent", {
            "loop_over": "items",
            "parallel_execution": True,
        })

        assert caps.loop_over == "items"
        assert caps.parallelizable is True


class TestGatherInputs:
    """Tests for gather_inputs function."""

    def test_gather_from_upstream_output(self):
        """Test gathering inputs from upstream node output."""
        incoming_edges = [
            {"from": "node_a", "data_mapping": {"output_key": "input_key"}}
        ]
        outputs_history = {
            "node_a": {"output_key": "value_from_a"}
        }

        result = gather_inputs(
            "node_b", incoming_edges, outputs_history, {}, {}
        )

        assert result.inputs["input_key"] == "value_from_a"
        assert not result.has_fatal_error

    def test_fallback_to_initial_inputs(self):
        """Test fallback to initial inputs when upstream doesn't have key."""
        incoming_edges = [
            {"from": "node_a", "data_mapping": {"missing_key": "input_key"}}
        ]
        outputs_history = {"node_a": {}}
        initial_inputs = {"missing_key": "from_initial"}

        result = gather_inputs(
            "node_b", incoming_edges, outputs_history, initial_inputs, {}
        )

        assert result.inputs["input_key"] == "from_initial"

    def test_upstream_error_creates_fatal_issue(self):
        """Test that upstream error creates a fatal issue."""
        incoming_edges = [
            {"from": "node_a", "data_mapping": {"key": "input"}}
        ]
        outputs_history = {"node_a": {"error": "Something failed"}}

        result = gather_inputs(
            "node_b", incoming_edges, outputs_history, {}, {}
        )

        assert result.has_fatal_error
        assert len(result.issues) == 1
        assert "Something failed" in result.issues[0].message

    def test_missing_optional_input_no_issue(self):
        """Test that missing optional inputs don't create issues."""
        incoming_edges = [
            {"from": "node_a", "data_mapping": {"missing": "optional_key"}}
        ]
        outputs_history = {"node_a": {}}

        result = gather_inputs(
            "node_b", incoming_edges, outputs_history, {},
            declared_output_keys={},
            optional_inputs={"optional_key"}
        )

        assert not result.has_fatal_error
        # No issues for optional inputs
        issues_for_optional = [i for i in result.issues if i.target_key == "optional_key"]
        assert len(issues_for_optional) == 0

    def test_multiple_edges_merged(self):
        """Test that inputs from multiple edges are merged."""
        incoming_edges = [
            {"from": "node_a", "data_mapping": {"out_a": "in_a"}},
            {"from": "node_b", "data_mapping": {"out_b": "in_b"}},
        ]
        outputs_history = {
            "node_a": {"out_a": "value_a"},
            "node_b": {"out_b": "value_b"},
        }

        result = gather_inputs(
            "node_c", incoming_edges, outputs_history, {}, {}
        )

        assert result.inputs["in_a"] == "value_a"
        assert result.inputs["in_b"] == "value_b"


class TestInjectContext:
    """Tests for inject_context function."""

    def test_inject_project_dir(self):
        """Test injecting project directory."""
        caps = AgentCapabilities(requires_project_dir=True)
        inputs = {"existing": "value"}

        result = inject_context(inputs, caps, project_base_output_dir="/output")

        assert result["project_base_output_dir"] == "/output"
        assert result["existing"] == "value"

    def test_inject_all_outputs(self):
        """Test injecting all agent outputs."""
        caps = AgentCapabilities(consumes_all_outputs=True)
        all_outputs = {"node_a": {"data": "value"}}

        result = inject_context({}, caps, all_agent_outputs=all_outputs)

        assert result["all_agent_outputs"] == all_outputs

    def test_no_injection_when_not_required(self):
        """Test that nothing is injected when not required."""
        caps = AgentCapabilities()
        inputs = {"key": "value"}

        result = inject_context(inputs, caps, project_base_output_dir="/output")

        assert "project_base_output_dir" not in result
        assert result == {"key": "value"}

    def test_original_inputs_not_modified(self):
        """Test that original inputs dict is not modified."""
        caps = AgentCapabilities(requires_project_dir=True)
        inputs = {"key": "value"}

        result = inject_context(inputs, caps, project_base_output_dir="/output")

        assert "project_base_output_dir" not in inputs
        assert "project_base_output_dir" in result


class TestAppendUpstreamIssue:
    """Tests for append_upstream_issue function."""

    def test_append_to_empty_inputs(self):
        """Test appending issue to empty inputs."""
        inputs = {}
        issue = UpstreamIssue(
            source_node="source",
            target_key="key",
            message="Error message",
            fatal=False,
        )

        append_upstream_issue(inputs, issue)

        assert "upstream_error_details" in inputs
        assert "upstream_error_messages" in inputs
        assert inputs["upstream_error_messages"] == ["Error message"]

    def test_fatal_issue_sets_error(self):
        """Test that fatal issue sets error key."""
        inputs = {}
        issue = UpstreamIssue(
            source_node="source",
            target_key="key",
            message="Fatal error",
            fatal=True,
        )

        append_upstream_issue(inputs, issue)

        assert inputs["error"] == "Fatal error"

    def test_non_fatal_does_not_set_error(self):
        """Test that non-fatal issue doesn't set error key."""
        inputs = {}
        issue = UpstreamIssue(
            source_node="source",
            target_key="key",
            message="Warning",
            fatal=False,
        )

        append_upstream_issue(inputs, issue)

        assert "error" not in inputs

    def test_empty_message_ignored(self):
        """Test that empty message is ignored."""
        inputs = {}
        issue = UpstreamIssue(
            source_node="source",
            target_key="key",
            message="",
            fatal=False,
        )

        append_upstream_issue(inputs, issue)

        assert "upstream_error_details" not in inputs
