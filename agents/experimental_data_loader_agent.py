import os
from .base_agent import Agent
from .registry import register_agent
from utils import log_status  # SCRIPT_DIR is no longer imported from here

@register_agent("ExperimentalDataLoaderAgent")
class ExperimentalDataLoaderAgent(Agent):
    def execute(self, inputs: dict) -> dict: # Type hint for dict
        current_system_message = self.get_formatted_system_message()
        if current_system_message.startswith("ERROR:"):
            return {"experimental_data_summary": "", "error": current_system_message}

        data_file_path = inputs.get("experimental_data_file_path")
        if not data_file_path:
            log_status(f"[{self.agent_id}] INFO: No experimental data file path provided. Proceeding without it.")
            return {"experimental_data_summary": "N/A - No experimental data file provided."}

        # Path resolution is now expected to be handled by the caller,
        # or data_file_path should be an absolute path.
        resolved_data_path = data_file_path
        log_status(f"[{self.agent_id}] INFO: Using experimental data path: '{resolved_data_path}'.")

        if not os.path.exists(resolved_data_path):
            log_status(f"[{self.agent_id}] WARNING: Experimental data file not found at '{resolved_data_path}'.")
            return {
                "experimental_data_summary": f"N/A - Experimental data file not found: {os.path.basename(resolved_data_path)}"}

        try:
            with open(resolved_data_path, 'r', encoding='utf-8') as f:
                data_content = f.read()

            if not data_content.strip():
                log_status(
                    f"[{self.agent_id}] WARNING: Experimental data file '{os.path.basename(resolved_data_path)}' is empty.")
                return {"experimental_data_summary": "N/A - Experimental data file is empty."}

            # Decide whether to use LLM for summarization or pass raw content
            if current_system_message and not current_system_message.startswith("ERROR:") and self.config_params.get("use_llm_to_summarize_data", False): # Example: Add a config flag
                max_exp_data_len = self.config_params.get("max_exp_data_len", 10000)
                truncated_data_content = data_content
                if len(data_content) > max_exp_data_len:
                    log_status(
                        f"[{self.agent_id}] INFO: Truncating experimental data from {len(data_content)} to {max_exp_data_len} for LLM processing.")
                    truncated_data_content = data_content[:max_exp_data_len]

                prompt = (
                    f"Please process and summarize the following experimental data content from file '"
                    f"{os.path.basename(resolved_data_path)}':\n\n---\n{truncated_data_content}\n---"
                )
                temperature = float(self.config_params.get("temperature", 0.6))
                summary = self.llm.complete(
                    system=current_system_message,
                    prompt=prompt,
                    model=self.model_name,
                    temperature=temperature,
                )

                if summary.startswith("Error:"):
                    log_status(
                        f"[{self.agent_id}] WARNING: Failed to summarize experimental data via LLM: {summary}. Passing raw content as fallback.")
                    return {"experimental_data_summary": data_content, # Fallback to raw content
                            "warning": f"LLM summary failed: {summary}. Raw data used."}
                log_status(f"[{self.agent_id}] INFO: Experimental data summarized by LLM.")
                return {"experimental_data_summary": summary}
            else:
                log_status(
                    f"[{self.agent_id}] INFO: Passing raw experimental data content as summary (LLM summarization not used or system message error).")
                return {"experimental_data_summary": data_content}
        except Exception as e:
            log_status(
                f"[{self.agent_id}] ERROR: Failed to read/process experimental data from {resolved_data_path}: {e}")
            return {"experimental_data_summary": "",
                    "error": f"Failed to read/process experimental data from {os.path.basename(resolved_data_path)}: {e}"}
