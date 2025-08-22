import os
from .base_agent import Agent
from .registry import register_agent
from utils import log_status

@register_agent("PDFSummarizerAgent")
class PDFSummarizerAgent(Agent):
    def execute(self, inputs: dict) -> dict: # Type hint for dict
        current_system_message = self.get_formatted_system_message()
        if current_system_message.startswith("ERROR:"):
            return {"summary": "", "error": current_system_message,
                    "original_pdf_path": inputs.get("original_pdf_path", "Unknown PDF")}
        pdf_text_content = inputs.get("pdf_text_content")
        original_pdf_path = inputs.get("original_pdf_path", "Unknown PDF")
        if inputs.get("pdf_text_content_error") or not pdf_text_content or (
                isinstance(pdf_text_content, str) and pdf_text_content.startswith("Error:")):
            error_msg = f"Invalid text content for summarization from {original_pdf_path}. Upstream error: {inputs.get('error', pdf_text_content)}"
            # log_status might be useful here if not already done upstream
            return {"summary": "", "error": error_msg, "original_pdf_path": original_pdf_path}
        max_len = self.config_params.get("max_input_length", 15000)
        if len(pdf_text_content) > max_len:
            log_status(
                f"[{self.agent_id}] INFO: Truncating PDF text from {len(pdf_text_content)} to {max_len} chars for summarization.")
            pdf_text_content = pdf_text_content[:max_len]
        prompt = f"Please summarize the following academic text from document '{os.path.basename(original_pdf_path)}':\n\n---\n{pdf_text_content}\n---"
        summary = self.llm.complete(
            system=current_system_message,
            prompt=prompt,
            model=self.model_name,
            temperature=0.6,
        )
        if summary.startswith("Error:"):
            return {"summary": "", "error": summary, "original_pdf_path": original_pdf_path}
        return {"summary": summary, "original_pdf_path": original_pdf_path}
