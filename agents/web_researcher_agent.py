"""Agent for performing simple web research summarization."""

from llm import LLMError
from utils import log_status

from .base_agent import Agent
from .registry import register_agent


@register_agent("WebResearcherAgent")
class WebResearcherAgent(Agent):
    """Use an LLM to generate a brief web research summary."""

    def execute(self, inputs: dict) -> dict:
        """Generate a web research summary using the provided context."""
        current_system_message = self.get_formatted_system_message()
        if current_system_message.startswith("ERROR:"):
            return {"web_summary": "", "error": current_system_message}

        cross_doc = inputs.get("cross_document_understanding")
        if not cross_doc:
            log_status(
                f"[{self.agent_id}] INFO: No 'cross_document_understanding' provided."
            )
            return {"web_summary": ""}

        prompt = (
            "Using the following cross-document understanding, perform brief web-style "
            "research and provide a concise summary with any additional relevant context:\n\n"
            f"{cross_doc}"
        )
        temperature = float(self.config_params.get("temperature", 0.6))
        try:
            summary = self.llm.complete(
                system=current_system_message,
                prompt=prompt,
                model=self.model_name,
                temperature=temperature,
            )
        except LLMError as e:  # pragma: no cover - LLM errors are rare
            return {"web_summary": "", "error": str(e)}

        log_status(
            f"[{self.agent_id}] INFO: Web research summary generated (length={len(summary)})."
        )
        return {"web_summary": summary}
