"""Tests for the helper function that invokes the OpenAI API."""

import llm_openai
from utils import call_openai_api, APP_CONFIG


def test_call_openai_api_passes_app_config(monkeypatch):
    """Ensure call_openai_api forwards configuration and parameters correctly."""
    # Prepare application configuration
    APP_CONFIG.clear()
    APP_CONFIG.update({
        "system_variables": {
            "openai_api_key": "test-key",
            "openai_api_timeout_seconds": 10,
        }
    })

    captured = {}

    class DummyLLM:  # pylint: disable=too-few-public-methods
        """Simple stand-in for the real OpenAI LLM class."""

        def __init__(self, app_config, api_key=None, timeout=0):
            """Record constructor arguments for later assertions."""
            captured["app_config"] = app_config
            captured["api_key"] = api_key
            captured["timeout"] = timeout

        def complete(self, system, prompt, model=None, temperature=None):
            """Record completion parameters and return a dummy response."""
            captured["system"] = system
            captured["prompt"] = prompt
            captured["model"] = model
            captured["temperature"] = temperature
            return "dummy-response"

    # Patch OpenAILLM to our dummy implementation
    monkeypatch.setattr(llm_openai, "OpenAILLM", DummyLLM)

    result = call_openai_api(
        prompt="hello",
        system_message="system-msg",
        model_name="test-model",
        temperature=0.3,
    )

    assert result == "dummy-response"
    assert captured["app_config"] is APP_CONFIG
    assert captured["api_key"] == "test-key"
    assert captured["timeout"] == 10
    assert captured["system"] == "system-msg"
    assert captured["prompt"] == "hello"
    assert captured["model"] == "test-model"
    assert captured["temperature"] == 0.3
