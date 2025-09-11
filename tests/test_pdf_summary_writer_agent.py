"""Tests for :class:`PDFSummaryWriterAgent`."""

import os
import shutil
import tempfile
import unittest

from agents.summary_writer_agent import PDFSummaryWriterAgent
from llm_fake import FakeLLM


class TestPDFSummaryWriterAgent(unittest.TestCase):
    """Ensure the summary writer agent persists summaries to disk."""

    def setUp(self):
        os.environ["OPENAI_API_KEY"] = "dummy"
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir)
        del os.environ["OPENAI_API_KEY"]

    def test_writes_files(self):
        summaries = [
            {"summary": "First", "original_pdf_path": "a.pdf"},
            {"summary": "Second", "original_pdf_path": "b.pdf"},
        ]
        app_config = {"system_variables": {"models": {}}, "agent_prompts": {}}
        fake = FakeLLM(app_config, response_map={})
        agent = PDFSummaryWriterAgent(
            "writer",
            "PDFSummaryWriterAgent",
            {"output_dir": self.temp_dir},
            fake,
            app_config,
        )
        result = agent.execute({"summaries_to_write": summaries})
        paths = result["written_summary_files"]
        self.assertEqual(len(paths), 2)
        for expected, path in zip(["First", "Second"], sorted(paths)):
            with open(path, "r", encoding="utf-8") as fh:
                self.assertEqual(fh.read(), expected)


if __name__ == "__main__":
    unittest.main()
