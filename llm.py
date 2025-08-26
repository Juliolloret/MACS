"""Interfaces and protocol definitions for LLM backends."""

from typing import Protocol, Optional, Dict, Any


class LLMError(Exception):
    """Represents a failure during an LLM completion call."""


class LLMClient(Protocol):
    """Protocol describing the minimum interface for LLM backends."""

    def __init__(
        self,
        app_config: Dict[str, Any],
        api_key: Optional[str] = None,
        timeout: int = 120,
    ) -> None:
        """Initialize the client.

        Parameters
        ----------
        app_config:
            Application configuration dictionary.
        api_key:
            Optional API key used for authentication.
        timeout:
            Maximum time in seconds to wait for network responses.
        """
        raise NotImplementedError

    def complete(  # pylint: disable=too-many-arguments
        self,
        *,
        system: str,
        prompt: str,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        extra: Optional[Dict] = None,
    ) -> str:
        """Perform a chat completion request and return the model output."""
        raise NotImplementedError

    def get_embeddings_client(self) -> Any:
        """Returns a cached embedding client if available."""
        raise NotImplementedError

    def close(self) -> None:
        """Releases any underlying resources held by the client."""
        raise NotImplementedError
