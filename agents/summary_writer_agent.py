"""Agent that writes individual PDF summaries to text files."""

import os
from typing import List, Dict

from .base_agent import Agent
from .registry import register_agent
from utils import log_status


@register_agent("PDFSummaryWriterAgent")
class PDFSummaryWriterAgent(Agent):
    """Persist each PDF summary to a separate text file."""

    def execute(self, inputs: dict) -> dict:  # type: ignore[override]
        """Write summaries from ``inputs`` to text files.

        Parameters
        ----------
        inputs: dict
            Expected to contain ``summaries_to_write`` which is a list of
            dictionaries with keys ``summary`` and ``original_pdf_path``.
        """
        summaries: List[Dict[str, str]] = inputs.get("summaries_to_write", [])
        output_dir = self.config_params.get("output_dir", "pdf_summaries")
        os.makedirs(output_dir, exist_ok=True)

        written_files = []
        for item in summaries:
            summary_text = item.get("summary", "")
            orig_path = item.get("original_pdf_path", "document.pdf")
            if not summary_text:
                continue
            base_name = os.path.splitext(os.path.basename(orig_path))[0]
            file_path = os.path.join(output_dir, f"{base_name}.txt")
            try:
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(summary_text)
                written_files.append(file_path)
                log_status(
                    f"[{self.agent_id}] INFO: Wrote summary for '{orig_path}' to '{file_path}'."
                )
            except OSError as e:
                log_status(
                    f"[{self.agent_id}] ERROR: Failed to write summary for '{orig_path}': {e}"
                )
        return {"written_summary_files": written_files}
