import json
import os
import time
from typing import Dict, Any, List

from .base_agent import Agent
from .registry import register_agent
from utils import log_status

@register_agent("ShortTermMemoryAgent")
class ShortTermMemoryAgent(Agent):
    """
    An agent that acts as a short-term memory buffer, collecting and structuring
    information from other agents within a single workflow run.
    """
    def __init__(self, agent_id, agent_type, config_params=None, llm=None, app_config=None):
        super().__init__(agent_id, agent_type, config_params, llm, app_config)

    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Collects a list of summaries and passes them through in a structured output.

        Args:
            inputs (dict): A dictionary expected to contain the key 'individual_summaries',
                           which holds a list of summary strings.

        Returns:
            dict: A dictionary containing the structured output under the key 'summaries_collection'.
                  Returns an error dictionary if the input is invalid.
        """
        log_status(f"[{self.agent_id}] INFO: Short-term memory agent is processing inputs.")

        individual_summaries = inputs.get("individual_summaries")

        if not isinstance(individual_summaries, list):
            error_msg = f"Input 'individual_summaries' is missing or not a list. Found: {type(individual_summaries).__name__}"
            log_status(f"[{self.agent_id}] ERROR: {error_msg}")
            return {"error": error_msg}

        log_status(f"[{self.agent_id}] INFO: Successfully collected {len(individual_summaries)} summaries.")

        return {"summaries_collection": individual_summaries}


@register_agent("LongTermMemoryAgent")
class LongTermMemoryAgent(Agent):
    """
    An agent that manages a persistent long-term memory, integrating new knowledge
    from workflow runs into a durable knowledge base.
    """
    def __init__(self, agent_id, agent_type, config_params=None, llm=None, app_config=None):
        super().__init__(agent_id, agent_type, config_params, llm, app_config)
        self.storage_filename_key = self.config_params.get("storage_filename_key")
        if not self.storage_filename_key:
            log_status(f"[{self.agent_id}] CRITICAL_WARNING: 'storage_filename_key' not configured. Long-term memory may not be persisted correctly.")

    def _get_storage_path(self, inputs: Dict[str, Any]) -> str:
        """Constructs the full storage path from the base directory and filename."""
        base_dir = inputs.get("project_base_output_dir")
        if not base_dir:
            log_status(f"[{self.agent_id}] WARNING: 'project_base_output_dir' not found in inputs. Defaulting to current directory for LTM.")
            base_dir = "."

        filename = self.app_config.get("system_variables", {}).get(self.storage_filename_key)
        if not filename:
            log_status(f"[{self.agent_id}] WARNING: Could not resolve filename from key '{self.storage_filename_key}'. Defaulting to 'ltm_fallback.json'.")
            filename = "ltm_fallback.json"

        return os.path.join(base_dir, filename)


    def _load_memory(self, storage_path: str) -> Dict[str, Any]:
        """Loads the long-term memory from the storage file."""
        if not storage_path or not os.path.exists(storage_path):
            log_status(f"[{self.agent_id}] INFO: No existing long-term memory file found at '{storage_path}'. Starting fresh.")
            return {}

        try:
            with open(storage_path, 'r', encoding='utf-8') as f:
                memory = json.load(f)
                log_status(f"[{self.agent_id}] INFO: Successfully loaded long-term memory from '{storage_path}'.")
                return memory
        except (json.JSONDecodeError, IOError) as e:
            log_status(f"[{self.agent_id}] ERROR: Failed to load or parse long-term memory from '{storage_path}': {e}. Starting with empty memory.")
            return {}

    def _save_memory(self, storage_path: str, memory_data: Dict[str, Any]):
        """Saves the long-term memory to the storage file."""
        if not storage_path:
            log_status(f"[{self.agent_id}] WARNING: No 'storage_path' provided. Skipping save.")
            return

        try:
            # Ensure the directory exists. Handles cases where the base_dir is nested.
            os.makedirs(os.path.dirname(storage_path), exist_ok=True)
            with open(storage_path, 'w', encoding='utf-8') as f:
                json.dump(memory_data, f, indent=4)
            log_status(f"[{self.agent_id}] INFO: Successfully saved updated long-term memory to '{storage_path}'.")
        except IOError as e:
            log_status(f"[{self.agent_id}] ERROR: Failed to save long-term memory to '{storage_path}': {e}.")

    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Loads existing memory, integrates new synthesis, saves the result, and outputs it.

        Args:
            inputs (dict): A dictionary expected to contain 'new_synthesis' and 'project_base_output_dir'.

        Returns:
            dict: A dictionary containing the 'updated_long_term_memory'.
        """
        log_status(f"[{self.agent_id}] INFO: Long-term memory agent is processing new knowledge.")

        storage_path = self._get_storage_path(inputs)

        new_synthesis = inputs.get("new_synthesis")
        if not new_synthesis:
            return {"error": "Input 'new_synthesis' was not provided."}

        # Load old memory
        existing_memory = self._load_memory(storage_path)
        existing_knowledge = existing_memory.get("knowledge_brief", "")

        if not existing_knowledge:
             # If there's no old memory, the new synthesis becomes the memory
            log_status(f"[{self.agent_id}] INFO: No existing knowledge. The new synthesis will become the initial long-term memory.")
            updated_knowledge = new_synthesis
        else:
            # Integrate new synthesis with existing knowledge using an LLM
            prompt = self.get_formatted_system_message()
            user_content = (
                f"EXISTING KNOWLEDGE BRIEF:\n---\n{existing_knowledge}\n---\n\n"
                f"NEWLY ADDED INFORMATION:\n---\n{new_synthesis}\n---\n\n"
                "Please provide the updated, integrated knowledge brief."
            )

            try:
                log_status(f"[{self.agent_id}] INFO: Calling LLM to integrate new knowledge.")
                response = self.llm.get_response(
                    system_message=prompt,
                    user_message=user_content,
                    model=self.model_name,
                    temperature=self.config_params.get("temperature", 0.5)
                )
                updated_knowledge = response
                log_status(f"[{self.agent_id}] INFO: LLM integration successful.")
            except Exception as e:
                error_msg = f"LLM call failed during knowledge integration: {e}"
                log_status(f"[{self.agent_id}] ERROR: {error_msg}")
                return {"error": error_msg}

        # Save the updated memory
        updated_memory_data = {"knowledge_brief": updated_knowledge, "last_updated": time.time()}
        self._save_memory(storage_path, updated_memory_data)

        return {"updated_long_term_memory": updated_knowledge}
