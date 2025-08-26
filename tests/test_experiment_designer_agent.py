"""Unit tests for the ExperimentDesignerAgent."""

import os
import unittest

from agents.experiment_designer_agent import ExperimentDesignerAgent
from llm_fake import FakeLLM


class TestExperimentDesignerAgent(unittest.TestCase):
    """Tests for ExperimentDesignerAgent functionality."""

    def setUp(self):
        """Create an agent instance and set required environment variables."""
        os.environ["OPENAI_API_KEY"] = "dummy_key"
        app_config = {"system_variables": {"models": {}}, "agent_prompts": {}}
        self.agent = ExperimentDesignerAgent(
            "eda",
            "ExperimentDesignerAgent",
            {},
            FakeLLM(app_config),
            app_config,
        )

    def tearDown(self):
        """Clean up environment variables after each test."""
        del os.environ["OPENAI_API_KEY"]

    def test_designs_experiment(self):
        """The agent returns a design for a valid hypothesis."""
        hypotheses = [{"hypothesis": "A", "justification": "Because"}]
        result = self.agent.execute({"hypotheses_list": hypotheses})
        designs = result["experiment_designs_list"]
        self.assertEqual(len(designs), 1)
        self.assertEqual(designs[0]["experiment_design"], "[FAKE] ok")
        self.assertNotIn("error", designs[0])

    def test_handles_upstream_error_flag(self):
        """The agent surfaces upstream errors when flagged in the input."""
        result = self.agent.execute({"hypotheses_list_error": True})
        self.assertEqual(result["experiment_designs_list"], [])
        self.assertIn("error", result)

    def test_invalid_hypothesis_object(self):
        """Invalid hypothesis objects result in an error entry."""
        result = self.agent.execute({"hypotheses_list": ["bad"]})
        designs = result["experiment_designs_list"]
        self.assertEqual(designs[0]["experiment_design"], "")
        self.assertIn("error", designs[0])


if __name__ == "__main__":
    unittest.main()
