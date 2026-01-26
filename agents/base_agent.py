"""Base functionality for MACS agents that utilise a language model."""

from __future__ import annotations

from typing import Optional, Dict, Any, List

from llm import LLMClient
from utils import get_model_name, get_prompt_text, log_status
from contracts import AgentCapabilities


class Agent:
    """Base class for all MACS agents that interact with an LLM.

    Subclasses can declare their capabilities by overriding the class-level
    capability attributes:
    - REQUIRES_PROJECT_DIR: Set to True if agent needs project_base_output_dir
    - CONSUMES_ALL_OUTPUTS: Set to True if agent needs all_agent_outputs
    - REQUIRED_INPUTS: List of input keys that must be provided
    - OPTIONAL_INPUTS: List of input keys that may be provided
    """

    # Capability declarations (override in subclasses)
    REQUIRES_PROJECT_DIR: bool = False
    CONSUMES_ALL_OUTPUTS: bool = False
    REQUIRED_INPUTS: List[str] = []
    OPTIONAL_INPUTS: List[str] = []

    def __init__(self, agent_id, agent_type, config_params=None, llm: LLMClient = None, app_config: Dict[str, Any] = None):  # pylint: disable=too-many-arguments,too-many-positional-arguments
        """Initialise a generic agent.

        Parameters
        ----------
        agent_id: Any
            Identifier used in the orchestration graph.
        agent_type: Any
            Human readable type name of the agent.
        config_params: dict | None
            Optional configuration dictionary for the agent.
        llm: LLMClient | None
            Instance implementing :class:`LLMClient` used to perform LLM calls.
        app_config: Dict[str, Any] | None
            Global application configuration.
        """
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

    def get_capabilities(self) -> AgentCapabilities:
        """Return the declared capabilities for this agent.

        Returns
        -------
        AgentCapabilities
            Dataclass containing capability flags and requirements.
        """
        return AgentCapabilities(
            requires_project_dir=self.REQUIRES_PROJECT_DIR,
            consumes_all_outputs=self.CONSUMES_ALL_OUTPUTS,
            needs_input_keys=list(self.REQUIRED_INPUTS),
            optional_input_keys=list(self.OPTIONAL_INPUTS),
            loop_over=self.config_params.get("loop_over"),
            parallelizable=self.config_params.get("parallel_execution", False),
        )

    def get_formatted_system_message(self, format_kwargs: Optional[Dict[str, Any]] = None) -> str:
        """Return the system message with optional string formatting."""
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

    def execute(self, _inputs: dict) -> dict:
        """Execute the agent's primary behaviour.

        Subclasses must override this method. The base implementation only
        reports configuration errors.
        """
        if self.system_message.startswith("ERROR:"):
            return {
                "error": f"Agent {self.agent_id} cannot execute due to configuration error for system message (key: '{self.config_params.get('system_message_key')}'): {self.system_message}"}
        raise NotImplementedError("Subclasses must implement execute()")
