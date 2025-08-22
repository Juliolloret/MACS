import os
import sys
import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from multi_agent_llm_system import GraphOrchestrator
from agents.base_agent import Agent
from agents.registry import register_agent
from llm_fake import FakeLLM

@register_agent("DummyAgent")
class DummyAgent(Agent):
    def execute(self, inputs: dict) -> dict:
        return {}


def test_topological_order():
    graph = {
        "nodes": [
            {"id": "A", "type": "DummyAgent"},
            {"id": "B", "type": "DummyAgent"},
            {"id": "C", "type": "DummyAgent"},
        ],
        "edges": [
            {"from": "A", "to": "B"},
            {"from": "B", "to": "C"},
            {"from": "A", "to": "C"},
        ],
    }
    orchestrator = GraphOrchestrator(graph, FakeLLM())
    order = orchestrator.node_order
    assert order.index("A") < order.index("B") < order.index("C")
