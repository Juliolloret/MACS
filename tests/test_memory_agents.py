"""Tests for short- and long-term memory agents."""

import os
import tempfile
import unittest

from agents.memory_agent import ShortTermMemoryAgent, LongTermMemoryAgent
from llm_fake import FakeLLM


class TestMemoryAgents(unittest.TestCase):
    """Verify the behavior of memory agents using a fake LLM."""

    def setUp(self):
        """Configure a fake LLM and environment for tests."""
        os.environ["OPENAI_API_KEY"] = "dummy_key"
        self.app_config = {
            "system_variables": {
                "models": {},
                "stm_path": "stm_store",
                "ltm_path": "ltm_store",
            },
            "agent_prompts": {},
        }

        class FakeLLMWithEmbeddings(FakeLLM):
            """Fake LLM providing deterministic embeddings for testing."""

            class DummyEmbeddings:
                """Simple embeddings client returning constant vectors."""

                def __call__(self, _text):
                    """Return a constant embedding regardless of input."""
                    return [0.1, 0.2, 0.3]

                def embed_documents(self, texts):
                    """Embed a list of documents."""
                    return [self(text) for text in texts]

                def embed_query(self, _text):
                    """Embed a single query."""
                    return [0.1, 0.2, 0.3]

            def get_embeddings_client(self):
                """Return the dummy embeddings client."""
                return self.DummyEmbeddings()

        self.llm = FakeLLMWithEmbeddings(self.app_config)

    def tearDown(self):
        """Remove test environment configuration."""
        del os.environ["OPENAI_API_KEY"]

    def test_short_term_memory_saves_vector_store(self):
        """Short term agent saves summaries and returns vector store path."""
        agent = ShortTermMemoryAgent(
            "stm",
            "ShortTermMemoryAgent",
            {"vector_store_path_key": "stm_path"},
            self.llm,
            self.app_config,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            summaries = [{"summary": "A"}, {"summary": "B"}]
            result = agent.execute(
                {"individual_summaries": summaries, "project_base_output_dir": tmpdir}
            )
            self.assertIn("vector_store_path", result)
            self.assertTrue(os.path.exists(result["vector_store_path"]))
            self.assertEqual(result["individual_summaries"], ["A", "B"])

    def test_short_term_memory_handles_invalid_input(self):
        """Short term agent handles invalid input gracefully."""
        agent = ShortTermMemoryAgent(
            "stm",
            "ShortTermMemoryAgent",
            {"vector_store_path_key": "stm_path"},
            self.llm,
            self.app_config,
        )
        result = agent.execute({"individual_summaries": "bad"})
        self.assertIsNone(result["vector_store_path"])
        self.assertEqual(result["individual_summaries"], [])

    def test_long_term_memory_updates_store(self):
        """Long term agent saves summaries to persistent storage."""
        agent = LongTermMemoryAgent(
            "ltm",
            "LongTermMemoryAgent",
            {"storage_filename_key": "ltm_path"},
            self.llm,
            self.app_config,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            summaries = ["A", "B"]
            result = agent.execute(
                {"individual_summaries": summaries, "project_base_output_dir": tmpdir}
            )
            self.assertIn("long_term_memory_path", result)
            self.assertTrue(os.path.exists(result["long_term_memory_path"]))

    def test_short_to_long_term_pipeline(self):
        """Pipeline from short-term to long-term memory operates correctly."""
        stm = ShortTermMemoryAgent(
            "stm",
            "ShortTermMemoryAgent",
            {"vector_store_path_key": "stm_path"},
            self.llm,
            self.app_config,
        )
        ltm = LongTermMemoryAgent(
            "ltm",
            "LongTermMemoryAgent",
            {"storage_filename_key": "ltm_path"},
            self.llm,
            self.app_config,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            stm_result = stm.execute(
                {"individual_summaries": [{"summary": "A"}], "project_base_output_dir": tmpdir}
            )
            ltm_result = ltm.execute(
                {
                    "individual_summaries": stm_result["individual_summaries"],
                    "project_base_output_dir": tmpdir,
                }
            )
            self.assertIn("long_term_memory_path", ltm_result)
            self.assertTrue(os.path.exists(ltm_result["long_term_memory_path"]))


if __name__ == "__main__":
    unittest.main()
