"""Tests for artifact_writer module."""

import os
import pytest
from unittest import mock

from artifact_writer import ArtifactWriter, write_consolidated_outputs


class TestArtifactWriter:
    """Tests for ArtifactWriter class."""

    def test_setup_output_directories(self, tmp_path):
        """Test that output directories are created."""
        system_vars = {
            "output_project_synthesis_folder_name": "synthesis",
            "output_project_hypotheses_folder_name": "hypotheses",
            "other_setting": "value",
        }

        writer = ArtifactWriter(str(tmp_path), system_vars)

        assert "synthesis" in writer.output_paths
        assert "hypotheses" in writer.output_paths
        assert os.path.exists(os.path.join(tmp_path, "synthesis"))
        assert os.path.exists(os.path.join(tmp_path, "hypotheses"))

    def test_write_file_success(self, tmp_path):
        """Test writing a file successfully."""
        writer = ArtifactWriter(
            str(tmp_path),
            {"output_project_test_folder_name": "test_folder"}
        )

        result = writer.write_file("test", "test.txt", "Hello World")

        assert result is True
        file_path = os.path.join(tmp_path, "test_folder", "test.txt")
        assert os.path.exists(file_path)
        with open(file_path, "r", encoding="utf-8") as f:
            assert f.read() == "Hello World"

    def test_write_file_empty_content(self, tmp_path):
        """Test writing with empty content creates placeholder."""
        writer = ArtifactWriter(
            str(tmp_path),
            {"output_project_test_folder_name": "test_folder"}
        )

        result = writer.write_file("test", "empty.txt", None)

        assert result is True
        file_path = os.path.join(tmp_path, "test_folder", "empty.txt")
        with open(file_path, "r", encoding="utf-8") as f:
            assert f.read() == "No content generated."

    def test_write_file_error_message(self, tmp_path):
        """Test writing with error message."""
        writer = ArtifactWriter(
            str(tmp_path),
            {"output_project_test_folder_name": "test_folder"}
        )

        result = writer.write_file("test", "error.txt", "", error="Something failed")

        assert result is True
        file_path = os.path.join(tmp_path, "test_folder", "error.txt")
        with open(file_path, "r", encoding="utf-8") as f:
            assert f.read() == "Error: Something failed"

    def test_write_file_missing_folder(self, tmp_path):
        """Test that writing to missing folder returns False."""
        writer = ArtifactWriter(str(tmp_path), {})

        result = writer.write_file("nonexistent", "test.txt", "content")

        assert result is False

    def test_get_folder_path(self, tmp_path):
        """Test getting folder path."""
        writer = ArtifactWriter(
            str(tmp_path),
            {"output_project_synthesis_folder_name": "synthesis_output"}
        )

        path = writer.get_folder_path("synthesis")

        assert path == os.path.join(tmp_path, "synthesis_output")

    def test_get_folder_path_missing(self, tmp_path):
        """Test getting path for missing folder."""
        writer = ArtifactWriter(str(tmp_path), {})

        path = writer.get_folder_path("nonexistent")

        assert path is None


class TestWriteConsolidatedOutputs:
    """Tests for write_consolidated_outputs function."""

    def test_writes_synthesis_outputs(self, tmp_path):
        """Test that synthesis outputs are written."""
        system_vars = {
            "output_project_synthesis_folder_name": "synthesis",
        }
        writer = ArtifactWriter(str(tmp_path), system_vars)

        outputs_history = {
            "multi_doc_synthesizer": {"multi_doc_synthesis_output": "Synthesis text"},
            "web_researcher": {"web_summary": "Web summary text"},
            "knowledge_integrator": {"integrated_knowledge_brief": "Brief text"},
        }

        write_consolidated_outputs(outputs_history, writer)

        synthesis_dir = os.path.join(tmp_path, "synthesis")
        assert os.path.exists(os.path.join(synthesis_dir, "multi_document_synthesis.txt"))
        assert os.path.exists(os.path.join(synthesis_dir, "web_research_summary.txt"))
        assert os.path.exists(os.path.join(synthesis_dir, "integrated_knowledge_brief.txt"))

    def test_writes_hypotheses_outputs(self, tmp_path):
        """Test that hypotheses outputs are written."""
        system_vars = {
            "output_project_hypotheses_folder_name": "hypotheses",
        }
        writer = ArtifactWriter(str(tmp_path), system_vars)

        outputs_history = {
            "hypothesis_generator": {
                "hypotheses_output_blob": '{"hypotheses": []}',
                "key_opportunities": "Opportunity 1",
                "hypotheses_list": ["Hypothesis 1", "Hypothesis 2"],
            },
        }

        write_consolidated_outputs(outputs_history, writer)

        hypo_dir = os.path.join(tmp_path, "hypotheses")
        assert os.path.exists(os.path.join(hypo_dir, "hypotheses_raw_llm_output.json"))
        assert os.path.exists(os.path.join(hypo_dir, "key_research_opportunities.txt"))
        assert os.path.exists(os.path.join(hypo_dir, "hypotheses_list.txt"))

        with open(os.path.join(hypo_dir, "hypotheses_list.txt"), "r") as f:
            content = f.read()
            assert "1. Hypothesis 1" in content
            assert "2. Hypothesis 2" in content

    def test_handles_empty_outputs(self, tmp_path):
        """Test graceful handling of empty outputs."""
        writer = ArtifactWriter(str(tmp_path), {})

        # Should not raise
        write_consolidated_outputs({}, writer)
