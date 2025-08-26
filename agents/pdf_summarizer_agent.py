"""Agent that produces summaries of PDF text content."""

import os

from utils import log_status
from llm import LLMError

from .base_agent import Agent
from .registry import register_agent

@register_agent("PDFSummarizerAgent")
class PDFSummarizerAgent(Agent):
    """Summarise the text extracted from a single PDF document."""

    def execute(self, inputs: dict) -> dict:  # Type hint for dict
        """Generate a concise summary for the PDF text in ``inputs``."""
        current_system_message = self.get_formatted_system_message()
        if current_system_message.startswith("ERROR:"):
            return {"summary": "", "error": current_system_message,
                    "original_pdf_path": inputs.get("original_pdf_path", "Unknown PDF")}
        pdf_text_content = inputs.get("pdf_text_content")
        original_pdf_path = inputs.get("original_pdf_path", "Unknown PDF")
        if inputs.get("pdf_text_content_error") or not pdf_text_content:
            error_msg = (
                f"Invalid text content for summarization from {original_pdf_path}. "
                f"Upstream error: {inputs.get('error', pdf_text_content)}"
            )
            return {"summary": "", "error": error_msg, "original_pdf_path": original_pdf_path}
        max_len = self.config_params.get("max_input_length", 15000)
        if len(pdf_text_content) > max_len:
            log_status(
                f"[{self.agent_id}] INFO: Truncating PDF text from {len(pdf_text_content)} to {max_len} chars for summarization.")
            pdf_text_content = pdf_text_content[:max_len]

        temperature = float(self.config_params.get("temperature", 0.6))
        prompt = (
            f"Please summarize the following academic text from document '"
            f"{os.path.basename(original_pdf_path)}':\n\n---\n{pdf_text_content}\n---"
        )
        try:
            summary = self.llm.complete(
                system=current_system_message,
                prompt=prompt,
                model=self.model_name,
                temperature=temperature,
            )
        except LLMError as e:
            return {"summary": "", "error": str(e), "original_pdf_path": original_pdf_path}
        return {"summary": summary, "original_pdf_path": original_pdf_path}
