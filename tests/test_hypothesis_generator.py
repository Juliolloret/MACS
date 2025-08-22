import json
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agents.hypothesis_generator_agent import HypothesisGeneratorAgent
from llm_fake import FakeLLM


def test_hypothesis_generator_parses_json():
    brief = "Some brief"
    num = 1
    prompt = (
        f"Based on the following 'Integrated Knowledge Brief':\n\n---\n{brief}\n---\n\n"
        f"Please provide your analysis, key research opportunities, and propose exactly {num} hypotheses strictly in the specified JSON format."
    )
    output = json.dumps({
        "key_opportunities": "Opportunity",
        "hypotheses": [{"hypothesis": "H1", "justification": "Because"}],
    })
    fake = FakeLLM({prompt: output})
    agent = HypothesisGeneratorAgent(
        "hypo",
        "HypothesisGeneratorAgent",
        {"num_hypotheses": num},
        llm=fake,
    )
    result = agent.execute({"integrated_knowledge_brief": brief})
    assert result["hypotheses_list"] == [{"hypothesis": "H1", "justification": "Because"}]
    assert result["key_opportunities"] == "Opportunity"
