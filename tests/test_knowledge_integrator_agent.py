"""Unit tests for the KnowledgeIntegratorAgent."""

import os
import unittest

from agents.knowledge_integrator_agent import KnowledgeIntegratorAgent
from llm_fake import FakeLLM


class TestKnowledgeIntegratorAgent(unittest.TestCase):
    """Tests for KnowledgeIntegratorAgent."""

    def setUp(self):
        """Create the agent and configure environment for tests."""
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
        """Clean up environment variables after tests."""
        del os.environ["OPENAI_API_KEY"]

    def test_integrates_sources(self):
        """Agent integrates multiple sources into a knowledge brief."""
        inputs = {
            "multi_doc_synthesis": "docs",
            "web_research_summary": "web",
            "experimental_data_summary": "data",
            "deep_research_summary": "deep",
            "long_term_memory": "memory",
        }
        result = self.agent.execute(inputs)
        self.assertIn("integrated_knowledge_brief", result)
        self.assertEqual(result["integrated_knowledge_brief"], "[FAKE] ok")
        self.assertNotIn("error", result)

    def test_handles_upstream_errors(self):
        """Agent handles upstream error flags without failing."""
        inputs = {
            "multi_doc_synthesis_error": True,
            "web_research_summary_error": True,
            "experimental_data_summary_error": True,
            "error": "fail",
            "deep_research_summary": "deep",
            "long_term_memory": "memory",
        }
        result = self.agent.execute(inputs)
        self.assertIn("integrated_knowledge_brief", result)
        self.assertEqual(result["integrated_knowledge_brief"], "[FAKE] ok")
        self.assertNotIn("error", result)

    def test_reports_missing_dependencies(self):
        """Agent reports missing mandatory dependency inputs."""
        result = self.agent.execute({"deep_research_summary_error": True})
        self.assertIn("error", result)
        self.assertIn("deep_research_summary", result["error"])
        self.assertIn("long_term_memory", result["error"])
        self.assertEqual(result["integrated_knowledge_brief"], "")


if __name__ == "__main__":
    unittest.main()
