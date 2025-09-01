import json
from storage import run_history as rh


def test_save_and_get_run(tmp_path, monkeypatch):
    run_history_path = tmp_path / "run_history.jsonl"
    run_config_dir = tmp_path / "configs"
    monkeypatch.setattr(rh, "RUN_HISTORY_PATH", str(run_history_path))
    monkeypatch.setattr(rh, "RUN_CONFIG_DIR", str(run_config_dir))

    run_id = "test-run"
    app_config = {"foo": "bar", "system_variables": {"prompt_ids": ["p1"]}}
    extra = {
        "pdf_file_paths": ["doc1.pdf"],
        "experimental_data_path": "data.csv",
        "project_base_output_dir": "out",
    }

    rh.save_run(run_id, app_config, extra)

    config_file = run_config_dir / f"{run_id}.json"
    assert config_file.exists()
    with config_file.open() as cf:
        assert json.load(cf) == app_config

    with run_history_path.open() as f:
        record = json.loads(f.readline())
    assert record["run_id"] == run_id
    assert record["config_path"] == f"configs/{run_id}.json"
    assert "config" not in record

    loaded = rh.get_run(run_id)
    assert loaded["config"] == app_config
    assert loaded["extra"] == extra

    all_records = rh.list_runs()
    assert all_records[0]["config_path"] == f"configs/{run_id}.json"
    assert "config" not in all_records[0]
