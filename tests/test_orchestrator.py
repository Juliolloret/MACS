import os
import sys
import unittest
import shutil

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from multi_agent_llm_system import GraphOrchestrator
from agents.base_agent import Agent
from agents.registry import register_agent
from llm_fake import FakeLLM

@register_agent("DummyAgent")
class DummyAgent(Agent):
    def execute(self, inputs: dict) -> dict:
        return {}

class TestGraphOrchestrator(unittest.TestCase):
    def setUp(self):
        self.test_outputs_dir = "test_outputs"
        if not os.path.exists(self.test_outputs_dir):
            os.makedirs(self.test_outputs_dir)

    def tearDown(self):
        if os.path.exists(self.test_outputs_dir):
            shutil.rmtree(self.test_outputs_dir)

    def test_topological_order(self):
        graph = {
            "nodes": [
                {"id": "A", "type": "DummyAgent", "config": {}},
                {"id": "B", "type": "DummyAgent", "config": {}},
                {"id": "C", "type": "DummyAgent", "config": {}},
            ],
            "edges": [
                {"from": "A", "to": "B"},
                {"from": "B", "to": "C"},
                {"from": "A", "to": "C"},
            ],
        }
        # A minimal app_config is needed for the agent initialization
        app_config = {
            "system_variables": {"models": {}},
            "agent_prompts": {}
        }
        orchestrator = GraphOrchestrator(graph, FakeLLM(app_config), app_config)
        order = orchestrator.node_order
        self.assertLess(order.index("A"), order.index("B"))
        self.assertLess(order.index("B"), order.index("C"))
