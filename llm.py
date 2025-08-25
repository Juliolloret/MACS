from typing import Protocol, Optional, Dict, Any


class LLMError(Exception):
    """Represents a failure during an LLM completion call."""


class LLMClient(Protocol):
    def __init__(self, app_config: Dict[str, Any], api_key: Optional[str] = None, timeout: int = 120):
        ...

    def complete(
        self,
        *,
        system: str,
        prompt: str,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        extra: Optional[Dict] = None,
    ) -> str:
        ...

    def get_embeddings_client(self) -> Any:
        """Returns a cached embedding client if available."""
        ...

    def close(self) -> None:
        """Releases any underlying resources held by the client."""
        ...
