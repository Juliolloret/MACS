"""Tests for validating the configuration schema of the orchestrator."""

import os
import unittest

from pydantic import ValidationError

from config_schema import validate_graph_definition
from llm_fake import FakeLLM


class TestConfigSchema(unittest.TestCase):
    """Ensure configuration schema errors are surfaced correctly."""

    def setUp(self):
        """Insert a dummy API key required by the orchestrator."""
        os.environ["OPENAI_API_KEY"] = "dummy"

    def tearDown(self):
        """Clean up environment variable after each test."""
        del os.environ["OPENAI_API_KEY"]

    def test_missing_node_type_raises(self):
        """Nodes without a type should trigger a validation error."""
        invalid_graph = {
            "nodes": [{"id": "a"}],
            "edges": []
        }
        with self.assertRaises(ValidationError):
            validate_graph_definition(invalid_graph)

    def test_invalid_data_mapping_type(self):
        """Edges with non-dict data_mapping should raise an error."""
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
            validate_graph_definition(invalid_graph)


if __name__ == "__main__":
    unittest.main()
