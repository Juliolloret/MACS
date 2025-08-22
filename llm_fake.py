from typing import Optional, Dict
from llm import LLMClient

class FakeLLM(LLMClient):
    def __init__(self, scripted_outputs: Optional[Dict[str, str]] = None):
        self.scripted = scripted_outputs or {}
        self.last_prompt = None
    def complete(self, *, system: str, prompt: str,
                 model: Optional[str] = None,
                 temperature: Optional[float] = None,
                 extra: Optional[Dict] = None) -> str:
        self.last_prompt = prompt
        return self.scripted.get(prompt, "[FAKE] ok")
