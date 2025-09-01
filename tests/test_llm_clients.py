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


def test_openai_llm_wraps_timeout_error(monkeypatch):
    """OpenAILLM.complete should wrap APITimeoutError in LLMError."""
    class TimeoutOpenAI:
        def __init__(self, *args, **kwargs):
            self.responses = self

        def create(self, **kwargs):  # pylint: disable=unused-argument
            raise APITimeoutError("timeout")

    monkeypatch.setattr(llm_openai, "OpenAI", TimeoutOpenAI)
    llm = OpenAILLM({}, api_key="test-key")

    with pytest.raises(LLMError):
        llm.complete(system="sys", prompt="hi", model="gpt-3")
