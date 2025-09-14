"""Tests for :class:`DeepResearchSummarizerAgent`."""

import os
import unittest

from agents.deep_research_summarizer_agent import DeepResearchSummarizerAgent
from llm_fake import FakeLLM


class TestDeepResearchSummarizerAgent(unittest.TestCase):
    """Unit tests for the deep research summarizer agent."""

    def setUp(self):
        """Ensure an API key is present for components that require it."""
        os.environ["OPENAI_API_KEY"] = "dummy_key"

    def tearDown(self):
        """Clean up the dummy API key set in ``setUp``."""
        del os.environ["OPENAI_API_KEY"]

    def _make_agent(self):
        app_config = {
            "system_variables": {"models": {}},
            "agent_prompts": {"deep_research_summarizer_sm": "You are a helpful summarizer."},
        }
        fake_llm = FakeLLM(app_config)
        config = {
            "model_key": "deep_research_summarizer_model",
            "system_message_key": "deep_research_summarizer_sm",
        }
        return DeepResearchSummarizerAgent(
            "test_agent",
            "DeepResearchSummarizerAgent",
            config,
            fake_llm,
            app_config,
        )

    def test_missing_user_query_returns_error_and_summary_key(self):
        """Agent returns an error and empty summary if ``user_query`` is absent."""
        agent = self._make_agent()
        result = agent.execute({})
        self.assertIn("deep_research_summary", result)
        self.assertIn("error", result)
        self.assertEqual(result["deep_research_summary"], "")


if __name__ == "__main__":
    unittest.main()
