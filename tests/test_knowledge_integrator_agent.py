import os
import unittest

from agents.knowledge_integrator_agent import KnowledgeIntegratorAgent
from llm_fake import FakeLLM


class TestKnowledgeIntegratorAgent(unittest.TestCase):
    def setUp(self):
        os.environ["OPENAI_API_KEY"] = "dummy_key"
        app_config = {"system_variables": {"models": {}}, "agent_prompts": {}}
        self.agent = KnowledgeIntegratorAgent(
            "kia",
            "KnowledgeIntegratorAgent",
            {},
            FakeLLM(app_config),
            app_config,
        )

    def tearDown(self):
        del os.environ["OPENAI_API_KEY"]

    def test_integrates_sources(self):
        inputs = {
            "multi_doc_synthesis": "docs",
            "web_research_summary": "web",
            "experimental_data_summary": "data",
        }
        result = self.agent.execute(inputs)
        self.assertIn("integrated_knowledge_brief", result)
        self.assertEqual(result["integrated_knowledge_brief"], "[FAKE] ok")
        self.assertNotIn("error", result)

    def test_handles_upstream_errors(self):
        inputs = {
            "multi_doc_synthesis_error": True,
            "web_research_summary_error": True,
            "experimental_data_summary_error": True,
            "error": "fail",
        }
        result = self.agent.execute(inputs)
        self.assertIn("integrated_knowledge_brief", result)
        self.assertEqual(result["integrated_knowledge_brief"], "[FAKE] ok")
        self.assertNotIn("error", result)


if __name__ == "__main__":
    unittest.main()
