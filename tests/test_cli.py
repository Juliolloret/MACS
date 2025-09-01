import json
import os
import subprocess
import sys
from pathlib import Path

from cli import _collect_pdf_paths

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_collect_pdf_paths_dedup(tmp_path):
    nested = tmp_path / "nested" / "inner"
    nested.mkdir(parents=True)
    file1 = nested / "file1.pdf"
    file1.write_text("content")
    file2 = tmp_path / "file2.PDF"
    file2.write_text("content")

    # Include both the parent and nested directories to create duplicates
    result = _collect_pdf_paths([str(tmp_path), str(nested)])
    assert sorted(result) == sorted({str(file1), str(file2)})


def test_cli_requires_arguments():
    completed = subprocess.run(
        [sys.executable, "-m", "cli"],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    assert "--pdf-dir and --out-dir are required" in completed.stdout
    assert completed.returncode == 0


def test_cli_list_plugins(tmp_path):
    agents_pkg = tmp_path / "agents"
    agents_pkg.mkdir()
    (agents_pkg / "__init__.py").write_text("")
    (agents_pkg / "registry.py").write_text(
        "def load_agents():\n    pass\n"
        "def load_plugins():\n    pass\n"
        "def list_plugins():\n"
        "    return [\n"
        "        {'name': 'plug1', 'version': '0.1', 'author': 'Alice'},\n"
        "        {'name': 'plug2', 'version': '0.2', 'author': ''},\n"
        "    ]\n"
    )

    # Raise if orchestration is accidentally executed
    (tmp_path / "multi_agent_llm_system.py").write_text(
        "def run_project_orchestration(*args, **kwargs):\n"
        "    raise AssertionError('orchestration should not run')\n"
    )

    env = {
        **os.environ,
        "PYTHONPATH": f"{tmp_path}{os.pathsep}{REPO_ROOT}",
    }

    completed = subprocess.run(
        [sys.executable, "-m", "cli", "--list-plugins"],
        capture_output=True,
        text=True,
        env=env,
        cwd=tmp_path,
    )
    assert "plug1 (0.1) by Alice" in completed.stdout
    assert "plug2 (0.2)" in completed.stdout
    assert completed.returncode == 0


def test_cli_adaptive(tmp_path):
    adaptive_pkg = tmp_path / "adaptive"
    adaptive_pkg.mkdir()
    (adaptive_pkg / "__init__.py").write_text("")
    called_file = tmp_path / "called.json"
    (adaptive_pkg / "adaptive_graph_runner.py").write_text(
        "import json, pathlib\n"
        "def adaptive_cycle(config_path, inputs, threshold, max_steps):\n"
        f"    pathlib.Path(r'{called_file}').write_text(json.dumps({{'config_path': config_path, 'inputs': inputs, 'threshold': threshold, 'max_steps': max_steps}}))\n"
    )

    (tmp_path / "multi_agent_llm_system.py").write_text(
        "def run_project_orchestration(*args, **kwargs):\n"
        "    raise AssertionError('orchestration should not run')\n"
    )

    pdf_dir = tmp_path / "pdfs"
    pdf_dir.mkdir()
    pdf_file = pdf_dir / "doc.pdf"
    pdf_file.write_text("data")

    out_dir = tmp_path / "out"
    out_dir.mkdir()

    env = {
        **os.environ,
        "PYTHONPATH": f"{tmp_path}{os.pathsep}{REPO_ROOT}",
    }

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "cli",
            "--pdf-dir",
            str(pdf_dir),
            "--out-dir",
            str(out_dir),
            "--adaptive",
        ],
        capture_output=True,
        text=True,
        env=env,
        cwd=tmp_path,
    )
    assert completed.returncode == 0

    data = json.loads(called_file.read_text())
    assert data["config_path"] == "config.json"
    assert data["inputs"]["initial_inputs"]["all_pdf_paths"] == [str(pdf_file)]
    assert data["inputs"]["initial_inputs"]["experimental_data_file_path"] == ""
    assert data["inputs"]["project_base_output_dir"] == str(out_dir)
    assert data["threshold"] == 1.0
    assert data["max_steps"] == 5
