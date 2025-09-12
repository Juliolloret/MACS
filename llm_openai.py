"""OpenAI-based LLM client with lightweight caching."""

from typing import Any, Dict, Optional
import json
import os

from cache import Cache, CachingEmbeddings
from llm import LLMClient, LLMError
from utils import get_model_name, log_status

try:
    from openai import (  # pylint: disable=import-error
        APIConnectionError,
        APITimeoutError,
        AuthenticationError,
        BadRequestError,
        OpenAI,
        RateLimitError,
    )
    OPENAI_AVAILABLE = True
except ImportError:  # pragma: no cover - handled in tests without openai
    OPENAI_AVAILABLE = False
    OpenAI = None  # type: ignore[invalid-name]
    APIConnectionError = APITimeoutError = RateLimitError = AuthenticationError = BadRequestError = Exception


class OpenAILLM(LLMClient):
    """LLM client backed by the official OpenAI Python SDK."""

    def __init__(self, app_config: Dict[str, Any], api_key: Optional[str] = None, timeout: int = 120):
        """Initialize the client and prepare response/embedding caches."""
        if not OPENAI_AVAILABLE:
            raise ImportError("openai library is required for OpenAILLM")
        self.app_config = app_config
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.timeout = timeout
        self._client = None
        self._embeddings_client = None
        # Simple JSON backed caches for completions and embeddings
        cache_dir = app_config.get("system_variables", {}).get("cache_dir", ".cache")
        try:
            os.makedirs(cache_dir, exist_ok=True)
        except OSError:
            # If directory can't be created, fall back to in-memory cache
            cache_dir = None
        responses_path = os.path.join(cache_dir, "llm_responses.json") if cache_dir else None
        embeds_path = os.path.join(cache_dir, "embeddings.json") if cache_dir else None
        self._response_cache = Cache(responses_path)
        self._embedding_cache = Cache(embeds_path)
        self.last_token_usage = 0
        self.total_tokens_used = 0

    @property
    def client(self):
        """Lazily create and return the underlying OpenAI client."""
        if self._client is None:
            self._client = OpenAI(api_key=self.api_key, timeout=self.timeout)
        return self._client

    def get_embeddings_client(self):
        """Return a caching embeddings client based on LangChain's wrapper."""
        if self._embeddings_client is None:
            try:
                from langchain_openai import OpenAIEmbeddings  # pylint: disable=import-error
            except ImportError as exc:  # pragma: no cover - optional dependency
                raise ImportError("langchain_openai library is required for embeddings") from exc
            base = OpenAIEmbeddings(client=self.client)
            self._embeddings_client = CachingEmbeddings(base, self._embedding_cache)
        return self._embeddings_client

    def complete(self, *, system: str, prompt: str,
                 model: Optional[str] = None,
                 temperature: Optional[float] = None,
                 extra: Optional[Dict] = None) -> str:  # pylint: disable=too-many-arguments, too-many-locals
        """Send a Responses API request to the OpenAI API."""
        chosen_model = model if model else get_model_name(self.app_config)
        sys_msg = system if system else "You are a helpful assistant."
        temp = temperature
        # GPT-5 only supports the default temperature of 1 and rejects explicit
        # values. To avoid "unsupported value" errors, omit the temperature
        # parameter entirely when targeting GPT-5 models.
        if chosen_model and chosen_model.startswith("gpt-5"):
            temp = None
        if chosen_model and "o4-mini" in chosen_model and temp is None:
            temp = 1.0
        if not self.api_key or self.api_key in [
            "YOUR_OPENAI_API_KEY_NOT_IN_CONFIG",
            "YOUR_ACTUAL_OPENAI_API_KEY",
            "KEY",
        ]:
            raise LLMError(f"OpenAI API key not configured for model {chosen_model}.")

        prompt_id = None
        conversation_id = None
        if extra:
            prompt_id = extra.get("prompt_id")
            conversation_id = extra.get("conversation_id")

        cache_key = self._response_cache.make_key(
            chosen_model, sys_msg, prompt, temp, prompt_id, conversation_id
        )
        cached = self._response_cache.get(cache_key)
        if cached is not None:
            log_status(f"[LLM] CACHE_HIT: Model='{chosen_model}'")
            return cached
        try:
            # Unify the system message and prompt into a single input string.
            # The Responses API expects a string, not a chat-style message list.
            combined_input = f"{sys_msg}\n\n{prompt}"

            params = {
                "model": chosen_model,
                "input": combined_input,
            }
            if temp is not None:
                params["temperature"] = temp
            if prompt_id:
                params["prompt_id"] = prompt_id
            if conversation_id:
                params["conversation_id"] = conversation_id

            response = self.client.responses.create(**params)
            usage = getattr(getattr(response, "usage", None), "total_tokens", 0)
            self.last_token_usage = usage
            self.total_tokens_used += usage

            try:
                # The modern Responses API provides output_text directly.
                if not hasattr(response, "output_text") or not response.output_text:
                    # Fallback for unexpected response format, though output_text is standard.
                    log_status(
                        f"[LLM] LLM_CALL_WARNING: Model='{chosen_model}' response has no output_text."
                    )
                    if not response.output or not response.output[0].content:
                        raise LLMError("Response object is missing expected content.")
                    # Legacy path: extract text from the first content block.
                    result = response.output[0].content[0].text.strip()
                else:
                    result = response.output_text.strip()
            except (AttributeError, IndexError) as e:
                log_status(
                    f"[LLM] LLM_CALL_ERROR: Model='{chosen_model}' could not parse response: {e}"
                )
                raise LLMError(
                    f"OpenAI API response for model {chosen_model} was malformed."
                ) from e

            snippet = result[:150].replace("\n", " ")
            log_status(
                f"[LLM] LLM_CALL_SUCCESS: Model='{chosen_model}', Response(start): '{snippet}...'"
            )
            self._response_cache.set(cache_key, result)
            return result
        except Exception as e:  # pragma: no cover - network errors not triggered in tests  # pylint: disable=broad-exception-caught
            err_name = type(e).__name__
            if isinstance(
                e,
                (
                    APIConnectionError,
                    APITimeoutError,
                    RateLimitError,
                    AuthenticationError,
                    BadRequestError,
                ),
            ):
                log_status(
                    f"[LLM] LLM_ERROR ({err_name}): API call with {chosen_model} failed: {e}"
                )
                detail = str(e)
                response_text = getattr(getattr(e, "response", None), "text", None)
                if response_text:
                    try:
                        err_json = json.loads(response_text)
                        detail = err_json.get("error", {}).get("message", detail)
                    except json.JSONDecodeError:
                        detail = response_text[:500]
                raise LLMError(
                    f"OpenAI API {err_name} for {chosen_model}: {detail}"
                ) from e
            log_status(
                f"[LLM] LLM_ERROR (General {err_name}): API call with {chosen_model} failed: {e}"
            )
            raise LLMError(
                f"API call with {chosen_model} failed ({err_name}): {e}"
            ) from e

    def get_last_token_usage(self) -> int:
        """Return the token usage of the most recent completion."""
        return self.last_token_usage

    def get_total_tokens_used(self) -> int:
        """Return the cumulative token usage for this client."""
        return self.total_tokens_used

    def close(self) -> None:
        """Release any resources held by the underlying SDK clients."""
        if self._client and hasattr(self._client, "close"):
            try:
                self._client.close()
            except Exception:  # pragma: no cover  # pylint: disable=broad-exception-caught
                pass
        self._client = None
        self._embeddings_client = None
