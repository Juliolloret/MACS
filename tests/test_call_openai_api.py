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

    captured = {"create_calls": 0}

    class DummyResponses:  # pylint: disable=too-few-public-methods
        def create(self, **kwargs):  # noqa: D401
            captured["create_calls"] += 1
            captured["temperature"] = kwargs.get("temperature")
            captured["model"] = kwargs.get("model")
            captured["input"] = kwargs.get("input")

            class Content:  # pylint: disable=too-few-public-methods
                text = "hi"

            class Output:  # pylint: disable=too-few-public-methods
                content = [Content()]

            class Usage:  # pylint: disable=too-few-public-methods
                total_tokens = 0

            class Resp:  # pylint: disable=too-few-public-methods
                output = [Output()]
                usage = Usage()

            return Resp()

    class DummyClient:  # pylint: disable=too-few-public-methods
        def __init__(self, api_key=None, timeout=None):
            self.responses = DummyResponses()

    class DummyCache:  # pylint: disable=too-few-public-methods
        store = {}

        def __init__(self, path=None):
            self.path = path

        def make_key(self, *args, **kwargs):  # noqa: D401
            return "key"

        def get(self, key):  # noqa: D401
            return type(self).store.get(key)

        def set(self, key, value):  # noqa: D401
            type(self).store[key] = value

    monkeypatch.setattr(llm_openai, "OpenAI", DummyClient)
    monkeypatch.setattr(llm_openai, "Cache", DummyCache)

    result1 = call_openai_api(
        prompt="hi",
        system_message="sys",
        model_name="gpt-5",
        temperature=0.4,
    )
    result2 = call_openai_api(
        prompt="hi",
        system_message="sys",
        model_name="gpt-5",
        temperature=0.4,
    )

    assert result1 == result2 == "hi"
    assert captured["temperature"] is None
    assert captured["model"] == "gpt-5"
    assert captured["input"] == [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "hi"},
    ]
    assert captured["create_calls"] == 1
