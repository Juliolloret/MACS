"""Agent that loads PDF files and extracts their raw text content."""

import os

from utils import log_status, PyPDF2  # PyPDF2 needed for PyPDF2.PdfReader

from .base_agent import Agent
from .registry import register_agent

@register_agent("PDFLoaderAgent")
class PDFLoaderAgent(Agent):
    """Extract raw text from a PDF document."""

    def execute(self, inputs: dict) -> dict:  # Type hint for dict matches common usage
        """Read the PDF specified in ``inputs`` and return its text content."""
        pdf_path = inputs.get("pdf_path")
        if not pdf_path:
            return {"pdf_text_content": "", "error": "PDF path not provided."}
        if not PyPDF2:
            return {"pdf_text_content": "", "error": "PyPDF2 library not available."}

        log_status(f"[{self.agent_id}] PDF_LOAD_START: Path='{pdf_path}'")
        if not os.path.exists(pdf_path):
            return {"pdf_text_content": "", "error": f"PDF file not found: {pdf_path}"}

        error = ""
        text_content = ""
        try:
            if os.path.getsize(pdf_path) == 0:
                error = f"PDF file is empty: {pdf_path}"
            else:
                with open(pdf_path, "rb") as pdf_file_obj:
                    pdf_reader = PyPDF2.PdfReader(pdf_file_obj)
                    if pdf_reader.is_encrypted and pdf_reader.decrypt("") == 0:
                        error = f"Failed to decrypt PDF: {os.path.basename(pdf_path)}."
                    else:
                        if pdf_reader.is_encrypted:
                            log_status(
                                f"[{self.agent_id}] PDF_LOAD_INFO: PDF '{os.path.basename(pdf_path)}' decrypted."
                            )
                        for page_obj in pdf_reader.pages:
                            text_content += page_obj.extract_text() or ""
                if not error and not text_content.strip():
                    log_status(
                        f"[{self.agent_id}] PDF_LOAD_WARNING: No text extracted from '{os.path.basename(pdf_path)}'."
                    )
        except (OSError, PyPDF2.errors.PdfReadError) as e:
            error = f"PDF extraction failed for {pdf_path}: {e}"

        if error:
            log_status(f"[{self.agent_id}] PDF_LOAD_ERROR: {error}")
            return {"pdf_text_content": "", "error": error}

        return {"pdf_text_content": text_content, "original_pdf_path": pdf_path}
