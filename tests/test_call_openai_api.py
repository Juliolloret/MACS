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


def test_call_openai_api_omits_temperature_for_gpt5(monkeypatch):
    """GPT-5 rejects explicit temperature values; ensure we don't send one."""
    APP_CONFIG.clear()
    APP_CONFIG.update({
        "system_variables": {
            "openai_api_key": "test-key",
            "openai_api_timeout_seconds": 10,
        }
    })

    captured = {}

    class DummyCompletions:  # pylint: disable=too-few-public-methods
        def create(self, **kwargs):  # noqa: D401
            captured["temperature"] = kwargs.get("temperature")

            class Message:  # pylint: disable=too-few-public-methods
                content = "hi"

            class Choice:  # pylint: disable=too-few-public-methods
                message = Message()

            class Resp:  # pylint: disable=too-few-public-methods
                choices = [Choice()]

            return Resp()

    class DummyClient:  # pylint: disable=too-few-public-methods
        def __init__(self, api_key=None, timeout=None):
            self.chat = type("Chat", (), {"completions": DummyCompletions()})()

    class DummyCache:  # pylint: disable=too-few-public-methods
        def __init__(self, path=None):
            pass

        def make_key(self, *args, **kwargs):  # noqa: D401
            return "key"

        def get(self, key):  # noqa: D401
            return None

        def set(self, key, value):  # noqa: D401
            pass

    monkeypatch.setattr(llm_openai, "OpenAI", DummyClient)
    monkeypatch.setattr(llm_openai, "Cache", DummyCache)

    call_openai_api(
        prompt="hi",
        system_message="sys",
        model_name="gpt-5",
        temperature=0.4,
    )

    assert captured["temperature"] is None
