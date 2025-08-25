import unittest
import os
import shutil
import time
from unittest.mock import patch, MagicMock
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter

from multi_agent_llm_system import GraphOrchestrator, load_app_config
from utils import APP_CONFIG
from llm_fake import FakeLLM
from agents.base_agent import Agent
from agents.registry import register_agent

@register_agent("A")
class AgentA(Agent):
    def execute(self, inputs):
        return {"out1": "output from A"}

@register_agent("B")
class AgentB(Agent):
    def execute(self, inputs):
        return {"out2": "output from B"}

@register_agent("C")
class AgentC(Agent):
    def execute(self, inputs):
        return {"out3": "output from C"}


@register_agent("SleepAgent")
class SleepAgent(Agent):
    def execute(self, inputs):
        duration = inputs.get("duration", 0.1)
        time.sleep(duration)
        return {"slept": duration}


class TestGraphOrchestrator(unittest.TestCase):

    def setUp(self):
        self.test_outputs_dir = "test_outputs"
        os.makedirs(self.test_outputs_dir, exist_ok=True)
        # The new agents require a dummy API key to be set, even for tests.
        os.environ["OPENAI_API_KEY"] = "dummy"
        APP_CONFIG.clear()

    def tearDown(self):
        if os.path.exists(self.test_outputs_dir):
            shutil.rmtree(self.test_outputs_dir)
        del os.environ["OPENAI_API_KEY"]
        APP_CONFIG.clear()

    def test_topological_sort(self):
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


    @patch('os.path.exists')
    @patch('agents.deep_research_summarizer_agent.FAISS')
    @patch('agents.memory_agent.FAISS')
    def test_full_graph_run_with_memory_agents(self, mock_faiss_memory, mock_faiss_deep_research, mock_os_path_exists):
        """
        Tests a full run of the graph defined in config.json with the new memory agents.
        Mocks FAISS to avoid file system and embedding issues.
        """
        # Configure the mock for os.path.exists
        mock_os_path_exists.return_value = True

        # Configure the mock for FAISS
        mock_vector_store = MagicMock()
        mock_faiss_memory.from_texts.return_value = mock_vector_store
        mock_faiss_deep_research.load_local.return_value = mock_vector_store
        mock_vector_store.similarity_search.return_value = [MagicMock(page_content="Relevant summary from search.")]

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
            "user_query": "What is the main takeaway from the documents?"
        }

        # Execute the orchestrator
        outputs_history = orchestrator.run(
            initial_inputs=initial_inputs,
            project_base_output_dir=self.test_outputs_dir
        )

        # Assertions
        self.assertIsNotNone(outputs_history)
        # Check that PDF loader ran without error
        self.assertNotIn("error", outputs_history.get("pdf_loader_node", {}).get("results", [{}])[0])
        # Check that memory agents ran without error
        self.assertNotIn("error", outputs_history.get("short_term_memory_node", {}))
        self.assertNotIn("error", outputs_history.get("long_term_memory_node", {}))
        # Check that the summarizer produced output
        self.assertIn("deep_research_summary", outputs_history.get("deep_research_summarizer", {}))
        self.assertNotIn("error", outputs_history.get("deep_research_summarizer", {}))

        # Verify that FAISS methods were called
        mock_faiss_memory.from_texts.assert_called_once()
        mock_vector_store.save_local.assert_called()
        mock_faiss_deep_research.load_local.assert_called_once()
        mock_vector_store.similarity_search.assert_called_once_with("What is the main takeaway from the documents?", k=3)

    def test_parallel_loop_execution(self):
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
        import builtins
        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name.startswith("graphviz"):
                raise ImportError
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=fake_import):
            path = orchestrator.visualize(output_base)
        self.assertTrue(path.endswith(".gv"))
        self.assertTrue(os.path.exists(path))

if __name__ == '__main__':
    unittest.main()
