"""Integration tests verifying agent collaboration."""

import os
import unittest

from agents.knowledge_integrator_agent import KnowledgeIntegratorAgent
from agents.hypothesis_generator_agent import HypothesisGeneratorAgent
from agents.experiment_designer_agent import ExperimentDesignerAgent
from agents.multi_doc_synthesizer_agent import MultiDocSynthesizerAgent
from llm_fake import FakeLLM


class TestAgentIntegration(unittest.TestCase):
    """Validate that core agents can run through an entire pipeline."""

    def setUp(self):
        """Prepare fake agents and configuration for testing."""
        os.environ["OPENAI_API_KEY"] = "dummy_key"
        self.app_config = {
            "system_variables": {"models": {}},
            "agent_prompts": {
                "hypothesis_generator_sm": "prompt",
                "multi_doc_synthesizer_sm": "prompt",
            },
        }
        self.llm = FakeLLM(self.app_config)
        self.multi_doc_agent = MultiDocSynthesizerAgent(
            "mds",
            "MultiDocSynthesizerAgent",
            {"system_message_key": "multi_doc_synthesizer_sm"},
            self.llm,
            self.app_config,
        )
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
        """Remove fake API key from environment."""
        del os.environ["OPENAI_API_KEY"]

    def test_pipeline(self):
        """Ensure each agent processes and passes data as expected."""
        summaries = [
            {"summary": "First", "original_pdf_path": "doc1.pdf"},
            {"summary": "Second", "original_pdf_path": "doc2.pdf"},
        ]
        multi_doc_out = self.multi_doc_agent.execute({"all_pdf_summaries": summaries})
        self.assertIn("multi_doc_synthesis_output", multi_doc_out)
        knowledge_out = self.knowledge_agent.execute(
            {
                "multi_doc_synthesis": multi_doc_out["multi_doc_synthesis_output"],
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
