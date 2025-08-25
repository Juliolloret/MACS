import unittest
import os
from agents.pdf_summarizer_agent import PDFSummarizerAgent
from llm_fake import FakeLLM

class TestPDFSummarizer(unittest.TestCase):
    def setUp(self):
        os.environ["OPENAI_API_KEY"] = "dummy_key"

    def tearDown(self):
        del os.environ["OPENAI_API_KEY"]

    def test_pdf_summarizer_truncates_input(self):
        long_text = "A" * 50
        truncated = long_text[:10]
        prompt = (
            f"Please summarize the following academic text from document 'doc.pdf':\n\n---\n"
            f"{truncated}\n---"
        )
        app_config = {
            "system_variables": {"models": {}},
            "agent_prompts": {"pdf_summarizer_sm": "You are an expert academic summarizer."}
        }
        fake = FakeLLM(app_config, response_map={prompt: "TRUNCATED"})
        agent = PDFSummarizerAgent(
            "test_agent",
            "PDFSummarizerAgent",
            {"max_input_length": 10, "temperature": 0.0, "model_key": "pdf_summarizer"},
            fake,
            app_config
        )
        result = agent.execute({"pdf_text_content": long_text, "original_pdf_path": "doc.pdf"})
        self.assertEqual(result["summary"], "TRUNCATED")

    def test_pdf_summarizer_happy_path(self):
        text = "This is a test."
        prompt = (
            f"Please summarize the following academic text from document 'doc.pdf':\n\n---\n"
            f"{text}\n---"
        )
        app_config = {
            "system_variables": {"models": {}},
            "agent_prompts": {"pdf_summarizer_sm": "You are an expert academic summarizer."}
        }
        fake = FakeLLM(app_config, response_map={prompt: "SUCCESS"})
        agent = PDFSummarizerAgent(
            "test_agent",
            "PDFSummarizerAgent",
            {"max_input_length": 100, "temperature": 0.0, "model_key": "pdf_summarizer"},
            fake,
            app_config
        )
        result = agent.execute({"pdf_text_content": text, "original_pdf_path": "doc.pdf"})
        self.assertEqual(result["summary"], "SUCCESS")

if __name__ == '__main__':
    unittest.main()
