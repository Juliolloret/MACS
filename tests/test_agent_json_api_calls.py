import json
from pathlib import Path

import llm_openai
from utils import APP_CONFIG, call_openai_api
import pytest

# Load JSON test cases describing agent prompts and models
TEST_CASES_PATH = Path(__file__).parent / "agent_openai_test_cases.json"
with open(TEST_CASES_PATH, "r", encoding="utf-8") as f:
    AGENT_TEST_CASES = json.load(f)


@pytest.mark.parametrize("case", AGENT_TEST_CASES)
def test_call_openai_api_from_json(monkeypatch, case):
    """Ensure each agent case from JSON triggers a proper API call."""
    APP_CONFIG.clear()
    APP_CONFIG.update({
        "system_variables": {
            "openai_api_key": "test-key",
            "openai_api_timeout_seconds": 10,
        }
    })

    captured = {}

    class DummyLLM:  # pylint: disable=too-few-public-methods
        def __init__(self, app_config, api_key=None, timeout=0):  # noqa: D401
            captured["api_key"] = api_key
            captured["timeout"] = timeout

        def complete(self, system, prompt, model=None, temperature=None):  # noqa: D401
            captured["system"] = system
            captured["prompt"] = prompt
            captured["model"] = model
            captured["temperature"] = temperature
            return "ok"

    monkeypatch.setattr(llm_openai, "OpenAILLM", DummyLLM)

    result = call_openai_api(
        prompt=case["prompt"],
        system_message=case["system"],
        model_name=case["model"],
    )

    assert result == "ok"
    assert captured["system"] == case["system"]
    assert captured["prompt"] == case["prompt"]
    assert captured["model"] == case["model"]
    assert captured["api_key"] == "test-key"
    assert captured["timeout"] == 10


def test_json_covers_all_openai_models():
    """Verify the JSON includes every OpenAI model up to GPT-5."""
    expected_models = {
        "gpt-3.5-turbo",
        "gpt-4",
        "gpt-4-turbo",
        "gpt-4.1",
        "gpt-4.1-mini",
        "gpt-4o",
        "gpt-4o-mini",
        "gpt-5",
    }
    models_in_json = {case["model"] for case in AGENT_TEST_CASES}
    assert models_in_json == expected_models
