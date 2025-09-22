import json
import os
import tempfile
from pathlib import Path

import pytest

from agents.deep_research_summarizer_agent import DeepResearchSummarizerAgent
from multi_agent_llm_system import run_project_orchestration
from storage import run_history as rh

# Minimal PDF bytes reused from pdf loader tests
DUMMY_PDF_BYTES = (
    b"%PDF-1.1\n1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n2 0 obj\n"
    b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n3 0 obj\n<< /Type /Page \n"
    b"/Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R /Resources << /Font << "
    b"/F1 5 0 R >> >> >>\nendobj\n4 0 obj\n<< /Length 44 >>\nstream\nBT /F1 24 Tf "
    b"100 700 Td (This is a test.) Tj ET\nendstream\nendobj\n5 0 obj\n<< /Type /Font \n"
    b"/Subtype /Type1 /BaseFont /Helvetica >>\nendobj\nxref\n0 6\n0000000000 65535 f \n"
    b"0000000010 00000 n \n0000000053 00000 n \n0000000102 00000 n \n0000000211 00000 n \n"
    b"0000000290 00000 n \ntrailer\n<< /Root 1 0 R /Size 6 >>\nstartxref\n344\n%%EOF"
)


def _write_dummy_pdf(path: Path) -> None:
    path.write_bytes(DUMMY_PDF_BYTES)


def _base_config():
    return {
        "system_variables": {
            "openai_api_key": "dummy",
            "llm_client": "fake",
            "default_llm_model": "fake-model",
            "output_project_synthesis_folder_name": "synth",
            "output_project_hypotheses_folder_name": "hypo",
            "output_project_experiments_folder_name": "exp",
        },
        "graph_definition": {"nodes": [], "edges": []},
    }


def test_run_project_fake_llm_allows_empty_key(monkeypatch):
    """Fake client should run without requiring an OpenAI API key."""
    app_config = _base_config()
    app_config["system_variables"]["openai_api_key"] = ""

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        history_path = tmp_path / "run_history.jsonl"
        config_dir = tmp_path / "run_configs"
        monkeypatch.setattr(rh, "RUN_HISTORY_PATH", str(history_path))
        monkeypatch.setattr(rh, "RUN_CONFIG_DIR", str(config_dir))

        pdf_path = tmp_path / "paper.pdf"
        _write_dummy_pdf(pdf_path)
        output_dir = tmp_path / "output"

        result = run_project_orchestration(
            [str(pdf_path)], None, str(output_dir), lambda _m: None, app_config
        )

        assert "error" not in result
        assert result["run_id"]


def test_run_project_records_history(monkeypatch):
    """Successful orchestration records run metadata."""
    app_config = _base_config()
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        history_path = tmp_path / "run_history.jsonl"
        config_dir = tmp_path / "run_configs"
        monkeypatch.setattr(rh, "RUN_HISTORY_PATH", str(history_path))
        monkeypatch.setattr(rh, "RUN_CONFIG_DIR", str(config_dir))

        pdf_path = tmp_path / "paper.pdf"
        _write_dummy_pdf(pdf_path)
        output_dir = tmp_path / "output"

        result = run_project_orchestration(
            [str(pdf_path)], None, str(output_dir), lambda _m: None, app_config
        )
        run_id = result["run_id"]

        with history_path.open() as f:
            records = [json.loads(line) for line in f if line.strip()]
        record = next(r for r in records if r["run_id"] == run_id)
        assert record["config_path"] == f"run_configs/{run_id}.json"

        config_file = history_path.parent / record["config_path"]
        assert json.loads(config_file.read_text()) == app_config


def test_run_project_missing_pdf(monkeypatch):
    """Missing PDF path triggers an error."""
    app_config = _base_config()
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        history_path = tmp_path / "run_history.jsonl"
        config_dir = tmp_path / "run_configs"
        monkeypatch.setattr(rh, "RUN_HISTORY_PATH", str(history_path))
        monkeypatch.setattr(rh, "RUN_CONFIG_DIR", str(config_dir))

        missing_pdf = tmp_path / "missing.pdf"
        output_dir = tmp_path / "output"

        result = run_project_orchestration(
            [str(missing_pdf)], None, str(output_dir), lambda _m: None, app_config
        )
        assert "error" in result
        assert f"Input PDF not found" in result["error"]


def test_run_project_output_dir_failure(monkeypatch):
    """Failure to create output directory returns an error."""
    app_config = _base_config()
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        history_path = tmp_path / "run_history.jsonl"
        config_dir = tmp_path / "run_configs"
        monkeypatch.setattr(rh, "RUN_HISTORY_PATH", str(history_path))
        monkeypatch.setattr(rh, "RUN_CONFIG_DIR", str(config_dir))

        pdf_path = tmp_path / "paper.pdf"
        _write_dummy_pdf(pdf_path)
        output_dir = tmp_path / "output"

        original_makedirs = os.makedirs

        def fail_makedirs(path, exist_ok=False):  # pylint: disable=unused-argument
            if path == str(output_dir):
                raise OSError("nope")
            return original_makedirs(path, exist_ok=exist_ok)

        monkeypatch.setattr(os, "makedirs", fail_makedirs)

        result = run_project_orchestration(
            [str(pdf_path)], None, str(output_dir), lambda _m: None, app_config
        )
        assert "error" in result
        assert "Could not create project output directory" in result["error"]


def test_run_project_passes_user_query_to_deep_research(monkeypatch):
    """User queries are forwarded to the deep research agent, including defaults."""
    app_config = _base_config()
    app_config["system_variables"]["default_user_query"] = "Default question?"
    app_config["agent_prompts"] = {"deep_research_summarizer_sm": "System prompt."}
    app_config["graph_definition"] = {
        "nodes": [
            {"id": "initial_input_provider", "type": "InitialInputProvider"},
            {
                "id": "deep_research_summarizer",
                "type": "DeepResearchSummarizerAgent",
                "config": {"system_message_key": "deep_research_summarizer_sm"},
            },
        ],
        "edges": [
            {
                "from": "initial_input_provider",
                "to": "deep_research_summarizer",
                "data_mapping": {"user_query": "user_query"},
            }
        ],
    }

    captured_queries = []

    def fake_execute(self, inputs):  # pylint: disable=unused-argument
        query = inputs.get("user_query")
        captured_queries.append(query)
        return {"deep_research_summary": f"summary for {query}"}

    monkeypatch.setattr(DeepResearchSummarizerAgent, "execute", fake_execute)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        history_path = tmp_path / "run_history.jsonl"
        config_dir = tmp_path / "run_configs"
        monkeypatch.setattr(rh, "RUN_HISTORY_PATH", str(history_path))
        monkeypatch.setattr(rh, "RUN_CONFIG_DIR", str(config_dir))

        output_default = tmp_path / "output_default"
        result_default = run_project_orchestration(
            pdf_file_paths=[],
            experimental_data_path="",
            project_base_output_dir=str(output_default),
            status_update_callback=lambda _m: None,
            app_config=app_config,
        )

        assert captured_queries[-1] == "Default question?"
        assert (
            result_default["deep_research_summarizer"]["deep_research_summary"]
            == "summary for Default question?"
        )

        custom_query = "What is artificial intelligence?"
        output_custom = tmp_path / "output_custom"
        result_custom = run_project_orchestration(
            pdf_file_paths=[],
            experimental_data_path="",
            project_base_output_dir=str(output_custom),
            status_update_callback=lambda _m: None,
            app_config=app_config,
            user_query=custom_query,
        )

        assert captured_queries[-1] == custom_query
        assert (
            result_custom["deep_research_summarizer"]["deep_research_summary"]
            == f"summary for {custom_query}"
        )
