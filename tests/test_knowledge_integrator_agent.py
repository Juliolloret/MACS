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
        self.fake_llm = FakeLLM(app_config)
        self.agent = KnowledgeIntegratorAgent(
            "kia",
            "KnowledgeIntegratorAgent",
            {},
            self.fake_llm,
            app_config,
        )

    def tearDown(self):
        """Clean up environment variables after tests."""
        del os.environ["OPENAI_API_KEY"]

    def test_integrates_sources(self):
        """Agent integrates multiple sources into a knowledge brief."""
        inputs = {
            "multi_doc_synthesis": "docs",
            "deep_research_summary": "deep",
            "web_research_summary": "web",
            "experimental_data_summary": "data",
        }
        result = self.agent.execute(inputs)
        self.assertIn("integrated_knowledge_brief", result)
        self.assertEqual(result["integrated_knowledge_brief"], "[FAKE] ok")
        self.assertNotIn("error", result)
        self.assertEqual(len(result.get("knowledge_sections", [])), 4)
        self.assertEqual(result.get("contributing_agents"), [])
        self.assertEqual(result.get("agent_context_details"), [])

    def test_prompt_includes_upstream_error_details(self):
        """Error context from upstream agents is surfaced in the prompt."""
        inputs = {
            "multi_doc_synthesis": "",
            "deep_research_summary": "",
            "web_research_summary": "",
            "experimental_data_summary": "",
            "upstream_error_details": [
                {
                    "source": "multi_doc_synthesizer",
                    "target": "multi_doc_synthesis",
                    "message": "Multi-document synthesis was unavailable.",
                },
                {
                    "source": "web_researcher",
                    "target": "web_research_summary",
                    "message": "Web research skipped due to earlier failure.",
                },
            ],
            "upstream_error_messages": [
                "Multi-document synthesis was unavailable.",
                "Web research skipped due to earlier failure.",
            ],
        }
        result = self.agent.execute(inputs)
        self.assertIn("integrated_knowledge_brief", result)
        prompt_used = self.fake_llm.last_prompt or ""
        self.assertIn("Upstream issues detected", prompt_used)
        self.assertIn("multi_doc_synthesizer", prompt_used)
        self.assertIn("Multi-document synthesis was unavailable.", prompt_used)

    def test_handles_upstream_errors(self):
        """Agent handles upstream error flags without failing."""
        inputs = {
            "multi_doc_synthesis_error": True,
            "deep_research_summary_error": True,
            "web_research_summary_error": True,
            "experimental_data_summary_error": True,
            "multi_doc_synthesis_error_message": "fail docs",
            "deep_research_summary_error_message": "fail deep",
            "web_research_summary_error_message": "fail web",
            "experimental_data_summary_error_message": "fail data",
            "error": "fail",
        }
        result = self.agent.execute(inputs)
        self.assertIn("integrated_knowledge_brief", result)
        self.assertEqual(result["integrated_knowledge_brief"], "[FAKE] ok")
        self.assertNotIn("error", result)

    def test_aggregates_all_agent_outputs(self):
        """All-agent context is surfaced and tracked for downstream use."""
        inputs = {
            "multi_doc_synthesis": "docs",
            "multi_doc_synthesis_source": "multi_doc_synthesizer",
            "all_agent_outputs": {
                "pdf_summarizer_node": {"results": [{"summary": "A"}]},
                "web_researcher": {"web_summary": "context"},
            },
        }
        result = self.agent.execute(inputs)
        self.assertIn("knowledge_sections", result)
        titles = [section["title"] for section in result["knowledge_sections"]]
        self.assertIn("Aggregated upstream agent outputs", titles)
        contributing = set(result.get("contributing_agents", []))
        self.assertIn("pdf_summarizer_node", contributing)
        details = {item["agent_id"] for item in result.get("agent_context_details", [])}
        self.assertIn("pdf_summarizer_node", details)


if __name__ == "__main__":
    unittest.main()
