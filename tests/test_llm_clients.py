import pytest

from llm_fake import FakeLLM
from llm_openai import APITimeoutError, OpenAILLM
from llm import LLMError
import llm_openai


def test_fake_llm_uses_cache_without_incrementing_tokens():
    """Calling FakeLLM with the same prompt twice should use cache."""
    llm = FakeLLM()
    first = llm.complete("system", "hello world", "gpt-3")
    tokens_after_first = llm.get_total_tokens_used()
    second = llm.complete("system", "hello world", "gpt-3")
    assert second == first
    assert llm.get_last_token_usage() == 0
    assert llm.get_total_tokens_used() == tokens_after_first


def test_openai_llm_wraps_timeout_error(monkeypatch, tmp_path):
    """OpenAILLM.complete should wrap APITimeoutError in LLMError."""
    class TimeoutOpenAI:
        class Responses:  # pylint: disable=too-few-public-methods
            def create(self, **kwargs):  # noqa: D401
                raise APITimeoutError("timeout")

        def __init__(self, *args, **kwargs):
            self.responses = self.Responses()

    monkeypatch.setattr(llm_openai, "OpenAI", TimeoutOpenAI)
    app_config = {"system_variables": {"cache_dir": str(tmp_path)}}
    llm = OpenAILLM(app_config, api_key="test-key")

    with pytest.raises(LLMError):
        llm.complete(system="sys", prompt="hi", model="gpt-3")


def test_openai_llm_uses_output_text_when_content_missing(monkeypatch, tmp_path):
    """Response.output_text should be used if output content is absent."""

    class DummyOpenAI:
        class Responses:  # pylint: disable=too-few-public-methods
            class DummyResponse:  # pylint: disable=too-few-public-methods
                def __init__(self):
                    self.output = [type("Obj", (), {"content": []})()]
                    self.output_text = "Hello world"
                    self.usage = type("Usage", (), {"total_tokens": 5})()

            def create(self, **kwargs):  # noqa: D401
                return self.DummyResponse()

        def __init__(self, *args, **kwargs):
            self.responses = self.Responses()

    monkeypatch.setattr(llm_openai, "OpenAI", DummyOpenAI)
    app_config = {"system_variables": {"cache_dir": str(tmp_path)}}
    llm = OpenAILLM(app_config, api_key="test-key")

    result = llm.complete(system="sys", prompt="hi", model="gpt-3")
    assert result == "Hello world"
