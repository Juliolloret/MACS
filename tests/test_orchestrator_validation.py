import os
import json
import pytest

from multi_agent_llm_system import GraphOrchestrator
from llm_fake import FakeLLM
from agents.base_agent import Agent
from agents.registry import register_agent
from utils import APP_CONFIG


@register_agent("LLMAgent")
class LLMAgent(Agent):
    """Agent that calls the LLM to generate a response."""

    def execute(self, inputs):  # pylint: disable=unused-argument
        self.llm.complete("", "hello world", "test-model")
        return {"result": "ok"}


@register_agent("FailAgent")
class FailAgent(Agent):
    """Agent that raises an exception when executed."""

    def execute(self, inputs):  # pylint: disable=unused-argument
        raise RuntimeError("boom")


@pytest.fixture(autouse=True)
def _env_setup():
    """Ensure required environment and global config for each test."""
    os.environ["OPENAI_API_KEY"] = "dummy"
    APP_CONFIG.clear()
    yield
    del os.environ["OPENAI_API_KEY"]
    APP_CONFIG.clear()


def test_graph_orchestrator_validation_errors():
    """Invalid graph definitions should raise ``ValueError`` on init."""
    llm = FakeLLM()

    cyclic_graph = {
        "nodes": [
            {"id": "a", "type": "LLMAgent"},
            {"id": "b", "type": "LLMAgent"},
        ],
        "edges": [
            {"from": "a", "to": "b"},
            {"from": "b", "to": "a"},
        ],
    }
    with pytest.raises(ValueError):
        GraphOrchestrator(cyclic_graph, llm, {})

    undefined_edge_graph = {
        "nodes": [{"id": "a", "type": "LLMAgent"}],
        "edges": [{"from": "a", "to": "missing"}],
    }
    with pytest.raises(ValueError):
        GraphOrchestrator(undefined_edge_graph, llm, {})


def test_node_metrics_written(tmp_path):
    """Running a graph writes execution metrics for each node."""
    graph = {
        "nodes": [
            {"id": "initial_input_provider", "type": "InitialInputProvider"},
            {"id": "llm_node", "type": "LLMAgent"},
        ],
        "edges": [{"from": "initial_input_provider", "to": "llm_node"}],
    }
    llm = FakeLLM()
    orchestrator = GraphOrchestrator(graph, llm, {})
    out_dir = tmp_path / "run"
    orchestrator.run({"text": "hi"}, str(out_dir))

    metrics_path = out_dir / "node_metrics.json"
    assert metrics_path.exists()
    with metrics_path.open(encoding="utf-8") as f:
        metrics = json.load(f)

    assert "llm_node" in metrics
    assert metrics["llm_node"]["execution_time_sec"] >= 0
    assert metrics["llm_node"]["tokens_used"] > 0


def test_observer_reports_downstream_error(tmp_path):
    """Observer agent should report errors from downstream failures."""
    graph = {
        "nodes": [
            {"id": "initial_input_provider", "type": "InitialInputProvider"},
            {"id": "llm_node", "type": "LLMAgent"},
            {"id": "fail_node", "type": "FailAgent"},
            {"id": "observer", "type": "ObserverAgent"},
        ],
        "edges": [
            {"from": "initial_input_provider", "to": "llm_node"},
            {"from": "llm_node", "to": "fail_node"},
        ],
    }

    llm = FakeLLM()
    orchestrator = GraphOrchestrator(graph, llm, {})
    outputs = orchestrator.run({"text": "hi"}, str(tmp_path))

    observer_output = outputs["observer"]
    assert observer_output["errors_found"] is True
    assert "fail_node" in observer_output["errors"]
