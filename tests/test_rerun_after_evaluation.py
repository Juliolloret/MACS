import json
from pathlib import Path

from adaptive import rerun_after_evaluation


def test_update_database(tmp_path):
    db_path = tmp_path / "db.json"
    rerun_after_evaluation.update_database(db_path, {"score": 0.5})
    data = json.loads(db_path.read_text())
    assert data["experiments"][0]["score"] == 0.5
    rerun_after_evaluation.update_database(db_path, {"score": 0.7})
    data = json.loads(db_path.read_text())
    assert len(data["experiments"]) == 2


def test_rerun_with_evaluation(monkeypatch, tmp_path):
    calls = {}

    def fake_cycle(config, inputs, eval_fn, threshold, max_steps):
        calls["args"] = (config, inputs, eval_fn, threshold, max_steps)

    monkeypatch.setattr(rerun_after_evaluation, "adaptive_cycle", fake_cycle)

    config_path = tmp_path / "config.json"
    inputs_path = tmp_path / "inputs.json"
    evaluation_path = tmp_path / "evaluation.json"
    db_path = tmp_path / "db.json"

    config_path.write_text(json.dumps({"graph_definition": {}}))
    inputs_path.write_text(
        json.dumps({"initial_inputs": {}, "project_base_output_dir": "."})
    )
    evaluation_path.write_text(json.dumps({"score": 0.3}))

    rerun_after_evaluation.rerun_with_evaluation(
        str(config_path),
        str(inputs_path),
        str(evaluation_path),
        str(db_path),
        threshold=0.8,
        max_steps=2,
    )

    db_data = json.loads(db_path.read_text())
    assert db_data["experiments"][0]["score"] == 0.3
    args = calls["args"]
    assert args[0] == str(config_path)
    assert args[1] == {"initial_inputs": {}, "project_base_output_dir": "."}
    assert callable(args[2])
    assert args[3] == 0.8
    assert args[4] == 2
