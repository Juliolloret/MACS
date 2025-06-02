import os
from typing import Dict
from .base_agent import Agent
from utils import log_status, PyPDF2 # PyPDF2 needed for PyPDF2.PdfReader

class PDFLoaderAgent(Agent):
    def execute(self, inputs: dict) -> dict: # Type hint for dict matches common usage
        base_pre_check_result = super().execute(inputs)
        if base_pre_check_result: return base_pre_check_result
        pdf_path = inputs.get("pdf_path")
        if not pdf_path: return {"pdf_text_content": "", "error": "PDF path not provided."}
        # Ensure log_status is available or handled if this agent is truly standalone.
        log_status(f"[{self.agent_id}] PDF_LOAD_START: Path='{pdf_path}'")
        if not PyPDF2: return {"pdf_text_content": "", "error": "PyPDF2 library not available."}
        if not os.path.exists(pdf_path): return {"pdf_text_content": "", "error": f"PDF file not found: {pdf_path}"}
        try:
            if os.path.getsize(pdf_path) == 0: return {"pdf_text_content": "",
                                                       "error": f"PDF file is empty: {pdf_path}"}
        except OSError as oe:
            return {"pdf_text_content": "", "error": f"Could not access file for size check: {pdf_path}, {oe}"}
        text_content = ""
        try:
            with open(pdf_path, 'rb') as pdf_file_obj:
                pdf_reader = PyPDF2.PdfReader(pdf_file_obj)
                if pdf_reader.is_encrypted:
                    if pdf_reader.decrypt('') == 0:
                        log_status(
                            f"[{self.agent_id}] PDF_LOAD_ERROR: Failed to decrypt PDF '{os.path.basename(pdf_path)}'.")
                        return {"pdf_text_content": "",
                                "error": f"Failed to decrypt PDF: {os.path.basename(pdf_path)}."}
                    log_status(f"[{self.agent_id}] PDF_LOAD_INFO: PDF '{os.path.basename(pdf_path)}' decrypted.")
                for page_obj in pdf_reader.pages: text_content += page_obj.extract_text() or ""
            if not text_content.strip(): log_status(
                f"[{self.agent_id}] PDF_LOAD_WARNING: No text extracted from '{os.path.basename(pdf_path)}'.")
            return {"pdf_text_content": text_content, "original_pdf_path": pdf_path}
        except Exception as e:
            log_status(f"[{self.agent_id}] PDF_LOAD_ERROR: PDF extraction failed for {pdf_path}: {e}")
            return {"pdf_text_content": "", "error": f"PDF extraction failed for {pdf_path}: {e}"}
