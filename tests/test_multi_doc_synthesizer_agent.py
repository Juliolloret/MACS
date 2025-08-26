"""Tests for :class:`MultiDocSynthesizerAgent`."""

import os
import unittest
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agents.multi_doc_synthesizer_agent import MultiDocSynthesizerAgent
from llm_fake import FakeLLM

class TestMultiDocSynthesizerAgent(unittest.TestCase):
    """Unit tests for the multi-document synthesizer agent."""

    def setUp(self):
        """Set up test dependencies and environment."""
        os.environ["OPENAI_API_KEY"] = "dummy_key"
        self.app_config = {
            "system_variables": {"models": {}},
            "agent_prompts": {"multi_doc_synthesizer_sm": "prompt"},
        }
        self.llm = FakeLLM(self.app_config)
        self.agent = MultiDocSynthesizerAgent(
            "mds",
            "MultiDocSynthesizerAgent",
            {"system_message_key": "multi_doc_synthesizer_sm"},
            self.llm,
            self.app_config,
        )

    def tearDown(self):
        """Clean up the environment variables."""
        del os.environ["OPENAI_API_KEY"]

    def test_synthesizes_valid_summaries(self):
        """The agent combines valid summaries into a synthesis."""
        summaries = [
            {"summary": "First", "original_pdf_path": "doc1.pdf"},
            {"summary": "Second", "original_pdf_path": "doc2.pdf"},
        ]
        result = self.agent.execute({"all_pdf_summaries": summaries})
        self.assertIn("multi_doc_synthesis_output", result)
        self.assertEqual(result["multi_doc_synthesis_output"], "[FAKE] ok")
        self.assertNotIn("error", result)

    def test_handles_upstream_error(self):
        """The agent reports an upstream error."""
        result = self.agent.execute({"all_pdf_summaries_error": True, "error": "boom"})
        self.assertIn("error", result)
        self.assertEqual(result["error"], "Upstream error providing PDF summaries: boom")


if __name__ == "__main__":
    unittest.main()
