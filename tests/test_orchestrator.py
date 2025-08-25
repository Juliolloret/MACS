import os
import sys
import unittest
import shutil

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
from multi_agent_llm_system import GraphOrchestrator
from agents.base_agent import Agent
from agents.registry import register_agent
from llm_fake import FakeLLM
from utils import load_app_config
from agents import load_agents

# Ensure all agents are registered for the test
load_agents("agents")

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

    def test_full_graph_run_with_memory_agents(self):
        """
        Tests a full run of the graph defined in config.json with the new memory agents.
        """
        # Load the actual application config
        app_config = load_app_config()
        # Use a fake LLM for testing
        llm = FakeLLM(app_config)

        # Create a dummy PDF for the loader to process
        dummy_pdf_path = os.path.join(self.test_outputs_dir, "dummy_paper.pdf")
        with open(dummy_pdf_path, "w") as f:
            f.write("This is a dummy PDF.")

        # Set up the orchestrator with the real graph
        graph_def = app_config.get("graph_definition")
        orchestrator = GraphOrchestrator(graph_def, llm, app_config)

        # Define initial inputs for the run
        initial_inputs = {
            "all_pdf_paths": [dummy_pdf_path],
            "experimental_data_file_path": None,
        }

        # Execute the orchestrator
        outputs_history = orchestrator.run(
            initial_inputs=initial_inputs,
            project_base_output_dir=self.test_outputs_dir
        )

        # Assertions
        self.assertIsNotNone(outputs_history)
        self.assertNotIn("error", outputs_history.get("pdf_loader_node", {}))
        self.assertIn("short_term_memory_node", outputs_history)
        self.assertIn("long_term_memory_node", outputs_history)
        self.assertNotIn("error", outputs_history.get("long_term_memory_node", {}))

        # Check if the long-term memory file was created
        ltm_filename = app_config.get("system_variables", {}).get("long_term_memory_filename")
        expected_ltm_path = os.path.join(self.test_outputs_dir, ltm_filename)
        self.assertTrue(os.path.exists(expected_ltm_path), f"Long-term memory file not found at {expected_ltm_path}")

        # Check the content of the LTM file
        with open(expected_ltm_path, 'r') as f:
            ltm_data = json.load(f)
        self.assertIn("knowledge_brief", ltm_data)
        # The fake LLM returns a predictable response
        self.assertIn("Fake response for prompt", ltm_data["knowledge_brief"])
