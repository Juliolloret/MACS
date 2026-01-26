"""Tests for orchestration_service module."""

import os
import pytest
from unittest import mock

from orchestration_service import (
    OrchestrationService,
    WorkflowConfig,
    WorkflowResult,
    run_workflow_from_paths,
)


class TestWorkflowConfig:
    """Tests for WorkflowConfig dataclass."""

    def test_basic_config(self):
        """Test basic configuration."""
        config = WorkflowConfig(
            config_path="config.json",
            pdf_paths=["/path/to/file.pdf"],
            output_dir="/output",
            user_query="Test query",
        )

        assert config.config_path == "config.json"
        assert config.pdf_paths == ["/path/to/file.pdf"]
        assert config.output_dir == "/output"
        assert config.user_query == "Test query"

    def test_optional_fields(self):
        """Test optional fields have defaults."""
        config = WorkflowConfig(
            config_path="config.json",
            pdf_paths=[],
            output_dir="/output",
        )

        assert config.user_query == ""
        assert config.project_name is None


class TestWorkflowResult:
    """Tests for WorkflowResult dataclass."""

    def test_successful_result(self):
        """Test successful workflow result."""
        result = WorkflowResult(
            success=True,
            output_dir="/output",
            outputs={"node1": {"data": "value"}},
        )

        assert result.success is True
        assert result.errors == []

    def test_failed_result(self):
        """Test failed workflow result."""
        result = WorkflowResult(
            success=False,
            output_dir="/output",
            errors=["Error 1", "Error 2"],
        )

        assert result.success is False
        assert len(result.errors) == 2


class TestOrchestrationService:
    """Tests for OrchestrationService class."""

    def test_load_config(self, tmp_path):
        """Test loading configuration."""
        # Create a minimal config file
        config_path = tmp_path / "config.json"
        config_path.write_text('{"system_variables": {}, "graph_definition": {"nodes": []}}')

        service = OrchestrationService()
        config = service.load_config(str(config_path))

        assert config is not None
        assert "system_variables" in config

    def test_log_callback_used(self):
        """Test that log callback is stored."""
        messages = []
        service = OrchestrationService(log_callback=messages.append)

        assert service.log_callback is not None

    def test_get_metrics_empty(self):
        """Test get_metrics returns empty dict before run."""
        service = OrchestrationService()

        metrics = service.get_metrics()

        assert metrics == {}

    def test_get_summary_empty(self):
        """Test get_summary returns empty dict before run."""
        service = OrchestrationService()

        summary = service.get_summary()

        assert summary == {}


class TestRunWorkflow:
    """Tests for workflow running functionality."""

    def test_run_workflow_creates_output_dir(self, tmp_path):
        """Test that run_workflow creates the output directory."""
        config_path = tmp_path / "config.json"
        config_path.write_text('{"system_variables": {}, "graph_definition": {"nodes": []}}')

        output_dir = tmp_path / "output"

        with mock.patch("orchestration_service.run_project_orchestration") as mock_run:
            mock_run.return_value = {}

            service = OrchestrationService()
            config = WorkflowConfig(
                config_path=str(config_path),
                pdf_paths=[],
                output_dir=str(output_dir),
            )

            result = service.run_workflow(config)

        assert os.path.exists(output_dir)

    def test_run_workflow_handles_exception(self, tmp_path):
        """Test that run_workflow handles exceptions gracefully."""
        config_path = tmp_path / "config.json"
        config_path.write_text('{"system_variables": {}, "graph_definition": {"nodes": []}}')

        with mock.patch("orchestration_service.run_project_orchestration") as mock_run:
            mock_run.side_effect = RuntimeError("Test error")

            service = OrchestrationService()
            config = WorkflowConfig(
                config_path=str(config_path),
                pdf_paths=[],
                output_dir=str(tmp_path / "output"),
            )

            result = service.run_workflow(config)

        assert result.success is False
        assert "Test error" in result.errors[0]

    def test_run_workflow_collects_node_errors(self, tmp_path):
        """Test that node errors are collected."""
        config_path = tmp_path / "config.json"
        config_path.write_text('{"system_variables": {}, "graph_definition": {"nodes": []}}')

        with mock.patch("orchestration_service.run_project_orchestration") as mock_run:
            mock_run.return_value = {
                "node1": {"result": "ok"},
                "node2": {"error": "Something failed"},
            }

            service = OrchestrationService()
            config = WorkflowConfig(
                config_path=str(config_path),
                pdf_paths=[],
                output_dir=str(tmp_path / "output"),
            )

            result = service.run_workflow(config)

        assert result.success is False
        assert len(result.errors) == 1
        assert "node2" in result.errors[0]


class TestRunWorkflowFromPaths:
    """Tests for run_workflow_from_paths convenience function."""

    def test_runs_workflow(self, tmp_path):
        """Test the convenience function."""
        config_path = tmp_path / "config.json"
        config_path.write_text('{"system_variables": {}, "graph_definition": {"nodes": []}}')

        with mock.patch("orchestration_service.run_project_orchestration") as mock_run:
            mock_run.return_value = {}

            result = run_workflow_from_paths(
                config_path=str(config_path),
                pdf_paths=[],
                output_dir=str(tmp_path / "output"),
            )

        assert isinstance(result, WorkflowResult)
