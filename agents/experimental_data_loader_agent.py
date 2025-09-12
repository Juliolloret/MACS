"""Agent for loading and summarizing experimental data from files."""

import os

from utils import log_status
from llm import LLMError

from .base_agent import Agent
from .registry import register_agent


@register_agent("ExperimentalDataLoaderAgent")
class ExperimentalDataLoaderAgent(Agent):
    """Load and optionally summarise experimental data from a file."""

    def execute(self, inputs: dict) -> dict:  # Type hint for dict
        """Read experimental data and optionally summarise it using the LLM."""
        current_system_message = self.get_formatted_system_message()
        if current_system_message.startswith("ERROR:"):
            return {"experimental_data_summary": "", "error": current_system_message}

        data_file_path = inputs.get("experimental_data_file_path")
        if not data_file_path:
            log_status(
                f"[{self.agent_id}] INFO: No experimental data file path provided. Proceeding without it."
            )
            return {"experimental_data_summary": "N/A - No experimental data file provided."}

        # Path resolution is now expected to be handled by the caller,
        # or data_file_path should be an absolute path.
        resolved_data_path = data_file_path
        log_status(
            f"[{self.agent_id}] INFO: Using experimental data path: '{resolved_data_path}'."
        )

        if not os.path.exists(resolved_data_path):
            log_status(
                f"[{self.agent_id}] WARNING: Experimental data file not found at '{resolved_data_path}'."
            )
            return {
                "experimental_data_summary": (
                    "N/A - Experimental data file not found: "
                    f"{os.path.basename(resolved_data_path)}"
                )
            }

        try:
            with open(resolved_data_path, "r", encoding="utf-8") as file_handle:
                data_content = file_handle.read()

            result: dict

            if not data_content.strip():
                log_status(
                    f"[{self.agent_id}] WARNING: Experimental data file "
                    f"'{os.path.basename(resolved_data_path)}' is empty."
                )
                result = {
                    "experimental_data_summary": "N/A - Experimental data file is empty."
                }
            elif (
                current_system_message
                and not current_system_message.startswith("ERROR:")
                and self.config_params.get("use_llm_to_summarize_data", False)
            ):
                max_exp_data_len = self.config_params.get("max_exp_data_len", 10000)
                truncated_data_content = data_content
                if len(data_content) > max_exp_data_len:
                    log_status(
                        f"[{self.agent_id}] INFO: Truncating experimental data from {len(data_content)} "
                        f"to {max_exp_data_len} for LLM processing."
                    )
                    truncated_data_content = data_content[:max_exp_data_len]

                prompt = (
                    "Please process and summarize the following experimental data content from file "
                    f"'{os.path.basename(resolved_data_path)}':\n\n---\n{truncated_data_content}\n---"
                )
                temperature = float(self.config_params.get("temperature", 0.6))
                reasoning_effort = self.config_params.get("reasoning_effort")
                verbosity = self.config_params.get("verbosity")
                extra_params = {}
                if reasoning_effort:
                    extra_params["reasoning"] = {"effort": reasoning_effort}
                if verbosity:
                    extra_params["text"] = {"verbosity": verbosity}
                try:
                    summary = self.llm.complete(
                        system=current_system_message,
                        prompt=prompt,
                        model=self.model_name,
                        temperature=temperature,
                        extra=extra_params,
                    )
                    log_status(
                        f"[{self.agent_id}] INFO: Experimental data summarized by LLM."
                    )
                    result = {"experimental_data_summary": summary}
                except LLMError as e:
                    log_status(
                        f"[{self.agent_id}] WARNING: Failed to summarize experimental data via LLM: {e}. "
                        "Passing raw content as fallback."
                    )
                    result = {
                        "experimental_data_summary": data_content,
                        "warning": f"LLM summary failed: {e}. Raw data used.",
                    }
            else:
                log_status(
                    f"[{self.agent_id}] INFO: Passing raw experimental data content as summary (LLM "
                    "summarization not used or system message error)."
                )
                result = {"experimental_data_summary": data_content}
        except (OSError, UnicodeDecodeError) as e:
            log_status(
                f"[{self.agent_id}] ERROR: Failed to read/process experimental data from {resolved_data_path}: {e}"
            )
            result = {
                "experimental_data_summary": "",
                "error": (
                    "Failed to read/process experimental data from "
                    f"{os.path.basename(resolved_data_path)}: {e}"
                ),
            }

        return result

