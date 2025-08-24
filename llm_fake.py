from typing import Optional, Dict
from llm import LLMClient

from typing import Any

class FakeLLM(LLMClient):
    def __init__(self, app_config: Dict[str, Any], scripted_outputs: Optional[Dict[str, str]] = None, api_key: Optional[str] = None, timeout: int = 120):
        self.scripted = scripted_outputs or {}
        self.last_prompt = None
        self.app_config = app_config
    def complete(self, *, system: str, prompt: str,
                 model: Optional[str] = None,
                 temperature: Optional[float] = None,
                 extra: Optional[Dict] = None) -> str:
        self.last_prompt = prompt
        return self.scripted.get(prompt, "[FAKE] ok")
