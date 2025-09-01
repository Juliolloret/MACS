"""Tests for evaluation plugin loading and adaptive cycle aggregation."""

import json
import importlib.metadata as md

import pytest

from adaptive.evaluation import load_evaluation_plugins
from adaptive import adaptive_graph_runner


def test_load_evaluation_plugins_from_config_and_entry_points(monkeypatch):
    """Evaluators can be loaded from config paths and entry points."""

    class DummyEP:
        name = "ep_eval"

        def load(self):
            class EPEval:
                def evaluate(self, outputs, graph_def, threshold, step):
                    return 0.6

            return EPEval

    def fake_entry_points():
        return {"macs.evaluators": [DummyEP()]}

    monkeypatch.setattr(md, "entry_points", fake_entry_points)

    evaluators = load_evaluation_plugins(
        ["agent_plugins.sample_evaluator:SampleEvaluator"]
    )
    assert len(evaluators) == 2
    scores = [ev.evaluate({"score": 0.4}, {}, 0.5, 1) for ev in evaluators]
    assert pytest.approx(sum(scores) / len(scores)) == 0.5


def test_adaptive_cycle_aggregates_scores(monkeypatch, tmp_path):
    """Adaptive cycle averages scores from multiple evaluators."""

    class DummyEP:
        name = "ep_eval"

        def load(self):
            class EpEval:
                def evaluate(self, outputs, graph_def, threshold, step):
                    return 0.6

            return EpEval

    monkeypatch.setattr(
        md, "entry_points", lambda: {"macs.evaluators": [DummyEP()]}
    )

    config = {
        "graph_definition": {},
        "evaluation_plugins": ["agent_plugins.sample_evaluator:SampleEvaluator"],
    }
    config_path = tmp_path / "cfg.json"
    config_path.write_text(json.dumps(config))
    inputs = {"initial_inputs": {}, "project_base_output_dir": "."}

    class DummyOrchestrator:
        def __init__(self, graph_def, llm, config):
            pass

        def run(self, initial_inputs, project_base_output_dir):
            return {"score": 0.4}

    monkeypatch.setattr(
        adaptive_graph_runner, "GraphOrchestrator", DummyOrchestrator
    )
    monkeypatch.setattr(
        adaptive_graph_runner, "_create_llm_client", lambda cfg: object()
    )

    mutated = {"called": False}

    def fake_mutate(graph_def, step):
        mutated["called"] = True
        return {"step": step}

    monkeypatch.setattr(adaptive_graph_runner, "mutate_graph_definition", fake_mutate)

    adaptive_graph_runner.adaptive_cycle(
        str(config_path), inputs, threshold=0.5, max_steps=1
    )
    assert not mutated["called"]
