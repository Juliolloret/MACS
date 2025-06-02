import os
import json
from typing import Optional, Dict, Any

# Utility functions that Agent class depends on, now imported from utils.py
from utils import get_model_name, get_prompt_text, log_status, APP_CONFIG

# Placeholder for call_openai_api if it's directly used by base Agent.
# Currently, it's used by specific agent execute methods, which will import it from utils.py.
# from utils import call_openai_api


class Agent:
    def __init__(self, agent_id, agent_type, config_params=None):
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.config_params = config_params if config_params else {}
        model_key_from_config = self.config_params.get("model_key")
        self.model_name = get_model_name(model_key_from_config)
        if not self.model_name:
            log_status(f"[AgentInit] WARNING: Agent ID='{self.agent_id}' has an empty model name. This might lead to errors if a model is required.")

        system_message_key = self.config_params.get("system_message_key")
        self.system_message = get_prompt_text(system_message_key)
        description = self.config_params.get("description", "No description provided.")
        log_status(
            f"[AgentInit] Created Agent: ID='{self.agent_id}', Type='{self.agent_type}', PrimaryModel='{self.model_name}'. SystemMsgKey='{system_message_key}'. Desc: {description}")
        
        if self.system_message.startswith("ERROR:"):
            log_status(
                f"[AgentInit] CRITICAL_WARNING: System message for Agent ID='{self.agent_id}' (Key: '{system_message_key}') could not be resolved: {self.system_message}")
        elif system_message_key and not self.system_message and not self.system_message.startswith("ERROR:"): # Explicitly check it's not already an error
            log_status(
                f"[AgentInit] WARNING: System message for Agent ID='{self.agent_id}' (Key: '{system_message_key}') is empty and not an error string. The agent might not behave as expected if a system message is crucial.")
        # The original `elif system_message_key and not self.system_message:` might be redundant now or could be removed
        # if the above warning is considered sufficient for empty non-error messages.
        # For now, keeping it to see if there are edge cases it might catch, though it seems covered.
        elif system_message_key and not self.system_message: # This will now only catch cases where it's empty AND system_message.startswith("ERROR:") was false.
            log_status(
                f"[AgentInit] WARNING: System message for Agent ID='{self.agent_id}' (Key: '{system_message_key}') is empty.")

    def get_formatted_system_message(self, format_kwargs: Optional[Dict[str, Any]] = None) -> str:
        if self.system_message.startswith("ERROR:"):
            return self.system_message
        if format_kwargs:
            try:
                return self.system_message.format(**format_kwargs)
            except KeyError as e:
                log_status(
                    f"[{self.agent_id}] SYSTEM_MSG_FORMAT_ERROR: Missing key {e} for system message template. Using unformatted message. Template was: '{self.system_message[:200]}...'")
                return self.system_message
        return self.system_message

    def execute(self, inputs: dict) -> dict:
        # Check if model_key was provided in config but resolved to an empty model_name
        if not self.model_name and self.config_params.get("model_key"):
            error_msg = f"Agent {self.agent_id} has no valid model name configured (model_key: '{self.config_params.get('model_key')}'), but a model key was specified. Cannot execute LLM-dependent task."
            log_status(f"[{self.agent_id}] EXECUTION_ERROR: {error_msg}")
            return {"error": error_msg}

        if self.system_message.startswith("ERROR:"):
            return {
                "error": f"Agent {self.agent_id} cannot execute due to configuration error for system message (key: '{self.config_params.get('system_message_key')}'): {self.system_message}"}
        return None
