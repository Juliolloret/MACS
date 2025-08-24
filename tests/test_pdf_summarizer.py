import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agents.pdf_summarizer_agent import PDFSummarizerAgent
from llm_fake import FakeLLM


def test_pdf_summarizer_truncates_input():
    long_text = "A" * 50
    truncated = long_text[:10]
    prompt = (
        "Please summarize the following academic text from document 'doc.pdf':\n\n---\n"
        + truncated + "\n---"
    )
    app_config = {
        "system_variables": {"models": {}},
        "agent_prompts": {}
    }
    fake = FakeLLM(app_config, {prompt: "TRUNCATED"})
    agent = PDFSummarizerAgent(
        "summarizer",
        "PDFSummarizerAgent",
        {"max_input_length": 10},
        llm=fake,
        app_config=app_config,
    )
    result = agent.execute({"pdf_text_content": long_text, "original_pdf_path": "doc.pdf"})
    assert result["summary"] == "TRUNCATED"
