import os
import unittest

from agents.experiment_designer_agent import ExperimentDesignerAgent
from llm_fake import FakeLLM


class TestExperimentDesignerAgent(unittest.TestCase):
    def setUp(self):
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
        del os.environ["OPENAI_API_KEY"]

    def test_designs_experiment(self):
        hypotheses = [{"hypothesis": "A", "justification": "Because"}]
        result = self.agent.execute({"hypotheses_list": hypotheses})
        designs = result["experiment_designs_list"]
        self.assertEqual(len(designs), 1)
        self.assertEqual(designs[0]["experiment_design"], "[FAKE] ok")
        self.assertNotIn("error", designs[0])

    def test_handles_upstream_error_flag(self):
        result = self.agent.execute({"hypotheses_list_error": True})
        self.assertEqual(result["experiment_designs_list"], [])
        self.assertIn("error", result)

    def test_invalid_hypothesis_object(self):
        result = self.agent.execute({"hypotheses_list": ["bad"]})
        designs = result["experiment_designs_list"]
        self.assertEqual(designs[0]["experiment_design"], "")
        self.assertIn("error", designs[0])


if __name__ == "__main__":
    unittest.main()
