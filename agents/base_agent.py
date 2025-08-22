from typing import Optional, Dict, Any

from llm import LLMClient
from utils import get_model_name, get_prompt_text, log_status


class Agent:
    def __init__(self, agent_id, agent_type, config_params=None, llm: LLMClient = None, app_config: Dict[str, Any] = None):
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.config_params = config_params if config_params else {}
        if llm is None:
            raise ValueError("LLMClient instance must be provided to Agent")
        if app_config is None:
            raise ValueError("app_config dictionary must be provided to Agent")
        self.llm = llm
        self.app_config = app_config
        model_key_from_config = self.config_params.get("model_key")
        self.model_name = get_model_name(self.app_config, model_key_from_config)
        system_message_key = self.config_params.get("system_message_key")
        self.system_message = get_prompt_text(self.app_config, system_message_key)
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
