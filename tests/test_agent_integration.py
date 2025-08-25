import os
import unittest

from agents.knowledge_integrator_agent import KnowledgeIntegratorAgent
from agents.hypothesis_generator_agent import HypothesisGeneratorAgent
from agents.experiment_designer_agent import ExperimentDesignerAgent
from llm_fake import FakeLLM


class TestAgentIntegration(unittest.TestCase):
    def setUp(self):
        os.environ["OPENAI_API_KEY"] = "dummy_key"
        self.app_config = {
            "system_variables": {"models": {}},
            "agent_prompts": {"hypothesis_generator_sm": "prompt"},
        }
        self.llm = FakeLLM(self.app_config)
        self.knowledge_agent = KnowledgeIntegratorAgent(
            "kia",
            "KnowledgeIntegratorAgent",
            {},
            self.llm,
            self.app_config,
        )
        self.hypothesis_agent = HypothesisGeneratorAgent(
            "hga",
            "HypothesisGeneratorAgent",
            {"num_hypotheses": 1},
            self.llm,
            self.app_config,
        )
        self.experiment_agent = ExperimentDesignerAgent(
            "eda",
            "ExperimentDesignerAgent",
            {},
            self.llm,
            self.app_config,
        )

    def tearDown(self):
        del os.environ["OPENAI_API_KEY"]

    def test_pipeline(self):
        knowledge_out = self.knowledge_agent.execute(
            {
                "multi_doc_synthesis": "docs",
                "web_research_summary": "web",
                "experimental_data_summary": "data",
            }
        )
        self.assertIn("integrated_knowledge_brief", knowledge_out)

        hypo_out = self.hypothesis_agent.execute(knowledge_out)
        self.assertIn("hypotheses_list", hypo_out)
        self.assertEqual(len(hypo_out["hypotheses_list"]), 1)

        exp_out = self.experiment_agent.execute(hypo_out)
        designs = exp_out["experiment_designs_list"]
        self.assertEqual(len(designs), 1)
        self.assertEqual(designs[0]["experiment_design"], "[FAKE] ok")


if __name__ == "__main__":
    unittest.main()
