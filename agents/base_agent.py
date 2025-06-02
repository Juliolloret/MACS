import os
import json
from typing import Optional, Dict, Any

# Attempt to import utility functions that Agent class depends on.
# These will likely be moved to a utils.py later or passed as dependencies,
# but for now, this reflects their current location.
from multi_agent_llm_system import get_model_name, get_prompt_text, log_status, APP_CONFIG

# Placeholder for call_openai_api if it's directly used by base Agent,
# though it seems it's used by specific agent execute methods rather than the base.
# If Agent.execute or other base methods use it, it needs to be available.
# from multi_agent_llm_system import call_openai_api


class Agent:
    def __init__(self, agent_id, agent_type, config_params=None):
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.config_params = config_params if config_params else {}
        model_key_from_config = self.config_params.get("model_key")
        self.model_name = get_model_name(model_key_from_config)
        system_message_key = self.config_params.get("system_message_key")
        self.system_message = get_prompt_text(system_message_key)
        description = self.config_params.get("description", "No description provided.")
        log_status(
            f"[AgentInit] Created Agent: ID='{self.agent_id}', Type='{self.agent_type}', PrimaryModel='{self.model_name}'. SystemMsgKey='{system_message_key}'. Desc: {description}")
        if self.system_message.startswith("ERROR:"):
            log_status(
                f"[AgentInit] CRITICAL_WARNING: System message for Agent ID='{self.agent_id}' (Key: '{system_message_key}') could not be resolved: {self.system_message}")
        elif system_message_key and not self.system_message:
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
        if self.system_message.startswith("ERROR:"):
            return {
                "error": f"Agent {self.agent_id} cannot execute due to configuration error for system message (key: '{self.config_params.get('system_message_key')}'): {self.system_message}"}
        return None
