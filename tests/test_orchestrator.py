"""Tests for the :class:`GraphOrchestrator` and related agents."""

import unittest
import os
import shutil
import time
from unittest.mock import patch, MagicMock

try:
    from reportlab.pdfgen import canvas
    from reportlab.lib.pagesizes import letter
except ImportError:  # pragma: no cover - optional dependency
    canvas = None
    letter = None

import builtins
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from multi_agent_llm_system import GraphOrchestrator, load_app_config
from utils import APP_CONFIG
from llm_fake import FakeLLM
from agents.base_agent import Agent
from agents.registry import register_agent


@register_agent("A")
class AgentA(Agent):
    """Test agent A."""

    def execute(self, inputs):  # pylint: disable=unused-argument
        """Return a fixed output."""
        return {"out1": "output from A"}


@register_agent("B")
class AgentB(Agent):
    """Test agent B."""

    def execute(self, inputs):  # pylint: disable=unused-argument
        """Return a fixed output."""
        return {"out2": "output from B"}


@register_agent("C")
class AgentC(Agent):
    """Test agent C."""

    def execute(self, inputs):  # pylint: disable=unused-argument
        """Return a fixed output."""
        return {"out3": "output from C"}


@register_agent("SleepAgent")
class SleepAgent(Agent):
    """Agent that sleeps for a specified duration."""

    def execute(self, inputs):  # pylint: disable=unused-argument
        """Sleep for the given duration and return it."""
        duration = inputs.get("duration", 0.1)
        time.sleep(duration)
        return {"slept": duration}


@register_agent("FailAgent")
class FailAgent(Agent):
    """Agent that raises an exception when executed."""

    def execute(self, inputs):  # pylint: disable=unused-argument
        raise RuntimeError("failure")


@register_agent("FlakyAgent")
class FlakyAgent(Agent):
    """Agent that fails once before succeeding."""

    def __init__(self, *args, **kwargs):  # pylint: disable=unused-argument
        super().__init__(*args, **kwargs)
        self.attempts = 0

    def execute(self, inputs):  # pylint: disable=unused-argument
        self.attempts += 1
        if self.attempts < 2:
            raise RuntimeError("flaky")
        return {"out": "success"}


class TestGraphOrchestrator(unittest.TestCase):
    """Integration tests for the graph orchestrator."""

    def setUp(self):
        """Create output directory and set environment variables."""
        self.test_outputs_dir = "test_outputs"
        os.makedirs(self.test_outputs_dir, exist_ok=True)
        # The new agents require a dummy API key to be set, even for tests.
        os.environ["OPENAI_API_KEY"] = "dummy"
        APP_CONFIG.clear()

    def tearDown(self):
        """Remove generated files and clear environment variables."""
        if os.path.exists(self.test_outputs_dir):
            shutil.rmtree(self.test_outputs_dir)
        del os.environ["OPENAI_API_KEY"]
        APP_CONFIG.clear()

    def test_topological_sort(self):
        """The orchestrator executes nodes in topological order."""
        config = {
            "graph_definition": {
                "nodes": [
                    {"id": "a", "type": "A"},
                    {"id": "b", "type": "B"},
                    {"id": "c", "type": "C"},
                ],
                "edges": [
                    {"from": "a", "to": "b", "data_mapping": {"out1": "in1"}},
                    {"from": "b", "to": "c", "data_mapping": {"out2": "in2"}},
                ],
            }
        }
        llm = FakeLLM()
        app_config = {"system_variables": {"default_llm_model": "test_model"}}
        orchestrator = GraphOrchestrator(config["graph_definition"], llm, app_config)
        self.assertIsNotNone(orchestrator)

        expected_incoming = {
            "b": [{"from": "a", "to": "b", "data_mapping": {"out1": "in1"}}],
            "c": [{"from": "b", "to": "c", "data_mapping": {"out2": "in2"}}],
        }
        self.assertEqual(dict(orchestrator.incoming_edges_map), expected_incoming)

        outputs_history = orchestrator.run({}, self.test_outputs_dir)
        self.assertEqual(outputs_history["a"], {"out1": "output from A"})
        self.assertEqual(outputs_history["b"], {"out2": "output from B"})
        self.assertEqual(outputs_history["c"], {"out3": "output from C"})

    def test_visualize_graph(self):
        """Visualization creates a graph file."""
        config = {
            "graph_definition": {
                "nodes": [
                    {"id": "a", "type": "A"},
                    {"id": "b", "type": "B"},
                ],
                "edges": [
                    {"from": "a", "to": "b"},
                ],
            }
        }
        llm = FakeLLM()
        app_config = {"system_variables": {"default_llm_model": "test_model"}}
        orchestrator = GraphOrchestrator(config["graph_definition"], llm, app_config)
        output_base = os.path.join(self.test_outputs_dir, "graph")
        path = orchestrator.visualize(output_base)
        self.assertTrue(os.path.exists(path))


    @patch("os.path.exists")
    def test_full_graph_run_with_synthesizer(self, mock_os_path_exists):
        """Run the full graph with the synthesizer, mocking file system existence."""
        if canvas is None:
            self.skipTest("reportlab is required for this test")

        # Configure the mock for os.path.exists
        mock_os_path_exists.return_value = True

        # Load the actual application config
        app_config = load_app_config()
        self.assertEqual(APP_CONFIG, app_config)
        # Use a fake LLM for testing
        llm = FakeLLM(app_config)

        # Create a dummy PDF for the loader to process
        dummy_pdf_path = os.path.join(self.test_outputs_dir, "dummy_paper.pdf")
        c = canvas.Canvas(dummy_pdf_path, pagesize=letter)
        c.drawString(100, 750, "This is a dummy PDF for testing.")
        c.save()

        # Set up the orchestrator with the real graph
        graph_def = app_config.get("graph_definition")
        orchestrator = GraphOrchestrator(graph_def, llm, app_config)

        # Define initial inputs for the run
        initial_inputs = {
            "all_pdf_paths": [dummy_pdf_path],
            "experimental_data_file_path": None,
            "user_query": "What is the main takeaway from the documents?",
        }

        # Execute the orchestrator
        outputs_history = orchestrator.run(
            initial_inputs=initial_inputs, project_base_output_dir=self.test_outputs_dir
        )

        # Assertions
        self.assertIsNotNone(outputs_history)
        # Check that PDF loader ran without error
        self.assertNotIn(
            "error",
            outputs_history.get("pdf_loader_node", {}).get("results", [{}])[0],
        )
        # Check that the synthesizer produced output
        self.assertIn(
            "multi_doc_synthesis_output",
            outputs_history.get("multi_doc_synthesizer_node", {}),
        )
        self.assertNotIn("error", outputs_history.get("multi_doc_synthesizer_node", {}))

        # Check that the knowledge integrator received the synthesizer's output
        self.assertIn(
            "integrated_knowledge_brief",
            outputs_history.get("knowledge_integrator", {}),
        )
        self.assertNotIn("error", outputs_history.get("knowledge_integrator", {}))

    def test_parallel_loop_execution(self):
        """Parallel loop execution reduces total run time."""
        config = {
            "graph_definition": {
                "nodes": [
                    {
                        "id": "sleeper",
                        "type": "SleepAgent",
                        "config": {
                            "loop_over": "durations",
                            "loop_item_input_key": "duration",
                            "parallel_execution": True,
                        },
                    }
                ],
                "edges": [],
            }
        }
        llm = FakeLLM()
        app_config = {"system_variables": {"default_llm_model": "test_model"}}
        orchestrator = GraphOrchestrator(config["graph_definition"], llm, app_config)
        durations = [0.5, 0.5]
        start = time.time()
        outputs = orchestrator.run({"durations": durations}, self.test_outputs_dir)
        elapsed = time.time() - start
        self.assertLess(elapsed, sum(durations))
        self.assertEqual(len(outputs.get("sleeper", {}).get("results", [])), 2)

    def test_parallel_loop_execution_order(self):
        """Parallel loop execution preserves result order."""
        config = {
            "graph_definition": {
                "nodes": [
                    {
                        "id": "sleeper",
                        "type": "SleepAgent",
                        "config": {
                            "loop_over": "durations",
                            "loop_item_input_key": "duration",
                            "parallel_execution": True,
                        },
                    }
                ],
                "edges": [],
            }
        }
        llm = FakeLLM()
        app_config = {"system_variables": {"default_llm_model": "test_model"}}
        orchestrator = GraphOrchestrator(config["graph_definition"], llm, app_config)
        durations = [0.1, 0.2, 0.3]
        outputs = orchestrator.run({"durations": durations}, self.test_outputs_dir)
        results = outputs.get("sleeper", {}).get("results", [])
        self.assertEqual([r.get("slept") for r in results], durations)

    def test_visualize_graph_without_graphviz(self):
        """visualize() falls back to a Mermaid file when graphviz is unavailable."""
        config = {
            "graph_definition": {
                "nodes": [
                    {"id": "a", "type": "A"},
                    {"id": "b", "type": "B"},
                ],
                "edges": [
                    {"from": "a", "to": "b"},
                ],
            }
        }
        llm = FakeLLM()
        app_config = {"system_variables": {"default_llm_model": "test_model"}}
        orchestrator = GraphOrchestrator(config["graph_definition"], llm, app_config)
        output_base = os.path.join(self.test_outputs_dir, "graph_no_gv")
        with patch("multi_agent_llm_system.Digraph", None), patch(
            "multi_agent_llm_system.ExecutableNotFound", None
        ):
            path = orchestrator.visualize(output_base)
        self.assertTrue(path.endswith(".mmd"))
        self.assertTrue(os.path.exists(path))

    def test_failure_policy_continue_override(self):
        """Node-level policy override allows continuing after failure."""
        config = {
            "graph_definition": {
                "nodes": [
                    {"id": "a", "type": "A"},
                    {"id": "fail", "type": "FailAgent", "config": {"failure_policy": "continue"}},
                    {"id": "c", "type": "C"},
                ],
                "edges": [
                    {"from": "a", "to": "fail"},
                    {"from": "fail", "to": "c"},
                ],
            }
        }
        llm = FakeLLM()
        app_config = {"system_variables": {"default_llm_model": "test_model"}}
        orchestrator = GraphOrchestrator(
            config["graph_definition"], llm, app_config, failure_policy="abort"
        )
        outputs = orchestrator.run({}, self.test_outputs_dir)
        self.assertIn("fail", outputs)
        self.assertIn("c", outputs)

    def test_failure_policy_abort(self):
        """Global abort policy stops execution on error."""
        config = {
            "graph_definition": {
                "nodes": [
                    {"id": "a", "type": "A"},
                    {"id": "fail", "type": "FailAgent"},
                    {"id": "c", "type": "C"},
                ],
                "edges": [
                    {"from": "a", "to": "fail"},
                    {"from": "fail", "to": "c"},
                ],
            }
        }
        llm = FakeLLM()
        app_config = {"system_variables": {"default_llm_model": "test_model"}}
        orchestrator = GraphOrchestrator(
            config["graph_definition"], llm, app_config, failure_policy="abort"
        )
        outputs = orchestrator.run({}, self.test_outputs_dir)
        self.assertIn("fail", outputs)
        self.assertNotIn("c", outputs)

    def test_failure_policy_retry(self):
        """Retry policy re-executes nodes on failure."""
        config = {
            "graph_definition": {
                "nodes": [
                    {
                        "id": "flaky",
                        "type": "FlakyAgent",
                        "config": {"failure_policy": "retry", "retries": 1},
                    }
                ],
                "edges": [],
            }
        }
        llm = FakeLLM()
        app_config = {"system_variables": {"default_llm_model": "test_model"}}
        orchestrator = GraphOrchestrator(config["graph_definition"], llm, app_config)
        outputs = orchestrator.run({}, self.test_outputs_dir)
        self.assertEqual(outputs.get("flaky"), {"out": "success"})
        self.assertEqual(orchestrator.agents["flaky"].attempts, 2)

if __name__ == '__main__':
    unittest.main()
