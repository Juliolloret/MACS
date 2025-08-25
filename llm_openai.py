from typing import Optional, Dict
import os
import json
from llm import LLMClient, LLMError
from utils import log_status, get_model_name
from cache import Cache, CachingEmbeddings

try:
    from openai import OpenAI as OpenAIClient, APIConnectionError, APITimeoutError, RateLimitError, AuthenticationError, BadRequestError
    OPENAI_AVAILABLE = True
except Exception:  # pragma: no cover - handled in tests without openai
    OPENAI_AVAILABLE = False
    OpenAIClient = None
    APIConnectionError = APITimeoutError = RateLimitError = AuthenticationError = BadRequestError = Exception


from typing import Optional, Dict, Any

class OpenAILLM(LLMClient):
    def __init__(self, app_config: Dict[str, Any], api_key: Optional[str] = None, timeout: int = 120):
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
        except Exception:
            # If directory can't be created, fall back to in-memory cache
            cache_dir = None
        responses_path = os.path.join(cache_dir, "llm_responses.json") if cache_dir else None
        embeds_path = os.path.join(cache_dir, "embeddings.json") if cache_dir else None
        self._response_cache = Cache(responses_path)
        self._embedding_cache = Cache(embeds_path)

    @property
    def client(self):
        if self._client is None:
            self._client = OpenAIClient(api_key=self.api_key, timeout=self.timeout)
        return self._client

    def get_embeddings_client(self):
        if self._embeddings_client is None:
            from langchain_openai import OpenAIEmbeddings
            base = OpenAIEmbeddings(client=self.client)
            self._embeddings_client = CachingEmbeddings(base, self._embedding_cache)
        return self._embeddings_client

    def complete(self, *, system: str, prompt: str,
                 model: Optional[str] = None,
                 temperature: Optional[float] = None,
                 extra: Optional[Dict] = None) -> str:
        chosen_model = model if model else get_model_name(self.app_config)
        sys_msg = system if system else "You are a helpful assistant."
        temp = temperature
        if chosen_model and "o4-mini" in chosen_model and temp is None:
            temp = 1.0
        if not self.api_key or self.api_key in [
            "YOUR_OPENAI_API_KEY_NOT_IN_CONFIG",
            "YOUR_ACTUAL_OPENAI_API_KEY",
            "KEY",
        ]:
            raise LLMError(f"OpenAI API key not configured for model {chosen_model}.")
        cache_key = self._response_cache.make_key(chosen_model, sys_msg, prompt, temp)
        cached = self._response_cache.get(cache_key)
        if cached is not None:
            log_status(f"[LLM] CACHE_HIT: Model='{chosen_model}'")
            return cached
        try:
            response = self.client.chat.completions.create(
                model=chosen_model,
                messages=[{"role": "system", "content": sys_msg}, {"role": "user", "content": prompt}],
                temperature=temp,
            )
            if not response.choices:
                log_status(
                    f"[LLM] LLM_CALL_ERROR: Model='{chosen_model}' response has no choices."
                )
                raise LLMError(
                    f"OpenAI API response had no choices for model {chosen_model}."
                )
            content = response.choices[0].message.content
            if not isinstance(content, str):
                log_status(
                    f"[LLM] LLM_CALL_ERROR_UNEXPECTED_CONTENT_TYPE: Model='{chosen_model}' returned content of type {type(content)}"
                )
                raise LLMError(
                    f"OpenAI API returned unexpected content type for model {chosen_model}."
                )
            result = content.strip()
            snippet = result[:150].replace('\n', ' ')
            log_status(f"[LLM] LLM_CALL_SUCCESS: Model='{chosen_model}', Response(start): '{snippet}...'")
            self._response_cache.set(cache_key, result)
            return result
        except Exception as e:  # pragma: no cover - network errors not triggered in tests
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
                if hasattr(e, "response") and hasattr(e.response, "text"):
                    try:
                        err_json = json.loads(e.response.text)
                        detail = err_json.get("error", {}).get("message", detail)
                    except Exception:
                        detail = e.response.text[:500]
                raise LLMError(
                    f"OpenAI API {err_name} for {chosen_model}: {detail}"
                )
            log_status(
                f"[LLM] LLM_ERROR (General {err_name}): API call with {chosen_model} failed: {e}"
            )
            raise LLMError(
                f"API call with {chosen_model} failed ({err_name}): {e}"
            )

    def close(self) -> None:
        if self._client and hasattr(self._client, "close"):
            try:
                self._client.close()
            except Exception:
                pass
        self._client = None
        self._embeddings_client = None
