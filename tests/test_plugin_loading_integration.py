import os
import pytest
from agents.registry import AGENT_REGISTRY, load_plugins
from multi_agent_llm_system import GraphOrchestrator
from llm_fake import FakeLLM


def test_plugin_agent_runs(tmp_path):
    os.environ["OPENAI_API_KEY"] = "dummy"
    plugin_dir = tmp_path / "plugin"
    plugin_dir.mkdir()
    (plugin_dir / "__init__.py").write_text("")
    plugin_code = """
from agents.base_agent import Agent
from agents.registry import AgentPlugin, PluginMetadata, MACS_VERSION

class EchoAgent(Agent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    def execute(self, inputs):
        return {"echo": "hello"}

PLUGIN = AgentPlugin(
    agent_class=EchoAgent,
    metadata=PluginMetadata(name='echo_agent', version='0.1', macs_version=MACS_VERSION),
)
"""
    (plugin_dir / "echo_plugin.py").write_text(plugin_code)
    prev_registry = AGENT_REGISTRY.copy()
    try:
        load_plugins(str(plugin_dir))
        config = {
            "graph_definition": {
                "nodes": [
                    {"id": "p", "type": "echo_agent"},
                ],
                "edges": [],
            }
        }
        llm = FakeLLM()
        app_config = {"system_variables": {"default_llm_model": "test_model"}}
        orchestrator = GraphOrchestrator(config["graph_definition"], llm, app_config)
        outputs = orchestrator.run({}, str(tmp_path))
        assert outputs["p"] == {"echo": "hello"}
    finally:
        AGENT_REGISTRY.clear()
        AGENT_REGISTRY.update(prev_registry)
        del os.environ["OPENAI_API_KEY"]
