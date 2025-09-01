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
    monkeypatch.setattr(multi_agent_llm_system, "Digraph", None)
    monkeypatch.setattr(multi_agent_llm_system, "ExecutableNotFound", None)
    output = tmp_path / "graph"
    orchestrator.visualize(str(output))
    dot_content = (tmp_path / "graph.gv").read_text()
    assert "1.23s, 45 tok" in dot_content
