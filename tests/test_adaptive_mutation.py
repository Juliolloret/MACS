"""Tests that adaptive_cycle mutates and saves graph definition when score low."""

import json
from pathlib import Path

import pytest

from adaptive import adaptive_graph_runner


class DummyEvaluator:
    """Evaluator returning constant low score to trigger mutation."""

    def evaluate(self, outputs, graph_def, threshold, step):
        return 0.0


def test_mutation_triggers_and_saves_graph(monkeypatch, tmp_path):
    """Mutation is called and config is saved after first iteration."""

    config = {"graph_definition": {"metadata": {}}, "evaluation_plugins": []}
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))
    inputs = {"initial_inputs": {}, "project_base_output_dir": str(tmp_path)}

    # Use dummy evaluator
    monkeypatch.setattr(
        adaptive_graph_runner, "load_evaluation_plugins", lambda cfg: [DummyEvaluator()]
    )

    # Avoid running real orchestrator or LLM
    class DummyOrchestrator:
        def __init__(self, graph_def, llm, config):
            pass

        def run(self, initial_inputs, project_base_output_dir):
            return {}

    monkeypatch.setattr(
        adaptive_graph_runner, "GraphOrchestrator", DummyOrchestrator
    )
    monkeypatch.setattr(
        adaptive_graph_runner, "_create_llm_client", lambda cfg: object()
    )

    mutate_calls = []

    def fake_mutate(graph_def, step):
        mutated = dict(graph_def)
        metadata = dict(mutated.get("metadata", {}))
        metadata["step"] = step
        mutated["metadata"] = metadata
        mutate_calls.append(step)
        return mutated

    monkeypatch.setattr(
        adaptive_graph_runner, "mutate_graph_definition", fake_mutate
    )

    saved_configs = []

    def fake_save_json(path, data):
        saved_configs.append(json.loads(json.dumps(data)))
        Path(path).write_text(json.dumps(data))

    monkeypatch.setattr(adaptive_graph_runner, "save_json", fake_save_json)

    adaptive_graph_runner.adaptive_cycle(
        str(config_path), inputs, threshold=1.0, max_steps=2
    )

    assert mutate_calls == [1, 2]
    assert saved_configs[0]["graph_definition"]["metadata"]["step"] == 1
