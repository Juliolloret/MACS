import os
import unittest

from pydantic import ValidationError

from multi_agent_llm_system import GraphOrchestrator
from llm_fake import FakeLLM


class TestConfigSchema(unittest.TestCase):
    def setUp(self):
        os.environ["OPENAI_API_KEY"] = "dummy"

    def tearDown(self):
        del os.environ["OPENAI_API_KEY"]

    def test_missing_node_type_raises(self):
        invalid_graph = {
            "nodes": [{"id": "a"}],
            "edges": []
        }
        with self.assertRaises(ValidationError):
            GraphOrchestrator(invalid_graph, FakeLLM(), {})

    def test_invalid_data_mapping_type(self):
        invalid_graph = {
            "nodes": [
                {"id": "a", "type": "A"},
                {"id": "b", "type": "B"},
            ],
            "edges": [
                {"from": "a", "to": "b", "data_mapping": ["not", "a", "dict"]}
            ]
        }
        with self.assertRaises(ValidationError):
            GraphOrchestrator(invalid_graph, FakeLLM(), {})


if __name__ == "__main__":
    unittest.main()
