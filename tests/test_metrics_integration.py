import json
import os

import pytest

from agents.base_agent import Agent
from agents.registry import register_agent
from event_subscribers import MetricsCollector
from events import EventBus
from llm_fake import FakeLLM
from multi_agent_llm_system import GraphOrchestrator
from utils import APP_CONFIG


@register_agent("MetricsTestAgent")
class MetricsTestAgent(Agent):
    """Agent that invokes the LLM to generate token usage."""

    def execute(self, _inputs):
        self.llm.complete(system="", prompt="hello world", model="test-model")
        return {"result": "ok"}


@pytest.fixture(autouse=True)
def _env_setup():
    """Ensure required environment and global config for each test."""
    os.environ["OPENAI_API_KEY"] = "dummy"
    APP_CONFIG.clear()
    yield
    del os.environ["OPENAI_API_KEY"]
    APP_CONFIG.clear()


def test_metrics_file_written_with_subscriber(tmp_path):
    """MetricsCollector writes output when attached to the EventBus."""
    graph = {
        "nodes": [
            {"id": "initial_input_provider", "type": "InitialInputProvider"},
            {"id": "metrics_node", "type": "MetricsTestAgent"},
        ],
        "edges": [{"from": "initial_input_provider", "to": "metrics_node"}],
    }

    llm = FakeLLM()
    metrics_path = tmp_path / "node_metrics.json"
    collector = MetricsCollector(output_path=str(metrics_path))
    bus = EventBus()
    bus.subscribe("*", collector.handle)

    orchestrator = GraphOrchestrator(graph, llm, {}, event_bus=bus)
    orchestrator.run({"text": "hi"}, str(tmp_path))

    assert metrics_path.exists()
    data = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert data["node_metrics"]["metrics_node"]["tokens_used"] > 0
