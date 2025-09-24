import os
import multi_agent_llm_system
from llm_fake import FakeLLM
from agents.base_agent import Agent
from agents.registry import register_agent

@register_agent("MetricAgent")
class MetricAgent(Agent):
    def execute(self, inputs):
        return {}

def test_visualize_includes_metrics(tmp_path, monkeypatch):
    config = {"nodes": [{"id": "a", "type": "MetricAgent"}], "edges": []}
    app_config = {"system_variables": {"default_llm_model": "test"}}
    orchestrator = multi_agent_llm_system.GraphOrchestrator(config, FakeLLM(), app_config)
    orchestrator.node_metrics["a"] = {"execution_time_sec": 1.23, "tokens_used": 45}

    captured_nodes = []

    class DummyNetwork:  # pylint: disable=too-few-public-methods
        def __init__(self, *args, **kwargs):  # pylint: disable=unused-argument
            return

        def barnes_hut(self):
            return None

        def add_node(self, *args, **kwargs):
            captured_nodes.append((args, kwargs))

        def add_edge(self, *args, **kwargs):  # pylint: disable=unused-argument
            return None

        def write_html(self, filename, notebook=False):  # pylint: disable=unused-argument
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            with open(filename, "w", encoding="utf-8") as handle:
                handle.write("<html></html>")

    monkeypatch.setattr(multi_agent_llm_system, "Network", DummyNetwork)
    monkeypatch.setattr(multi_agent_llm_system, "_open_graph_file", lambda path: None)

    output = tmp_path / "graph"
    orchestrator.visualize(str(output))

    assert any("1.23s, 45 tok" in kwargs.get("label", "") for _, kwargs in captured_nodes)
