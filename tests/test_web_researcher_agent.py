import pytest

from agents.web_researcher_agent import WebResearcherAgent
from llm_fake import FakeLLM


def test_web_researcher_produces_summary():
    app_config = {
        "agent_prompts": {"web_researcher_sm": "You are a web researcher."}
    }
    agent = WebResearcherAgent(
        agent_id="web_researcher",
        agent_type="WebResearcherAgent",
        config_params={"model_key": "pdf_summarizer", "system_message_key": "web_researcher_sm"},
        llm=FakeLLM(app_config=app_config),
        app_config=app_config,
    )
    result = agent.execute({"cross_document_understanding": "Some findings"})
    assert result["web_summary"], "Web research summary should not be empty"
