"""Observer agent for reviewing agent outputs and reporting errors."""

from typing import Dict

from utils import log_status

from .base_agent import Agent
from .registry import register_agent

@register_agent("ObserverAgent")
class ObserverAgent(Agent):
    """Reviews outputs from all agents and reports any errors found."""

    def execute(self, inputs: dict) -> dict:
        """Scan the full workflow history and surface any reported errors."""
        outputs_history: Dict[str, dict] = inputs.get("outputs_history", {})
        errors_found = {}
        for node_id, output in outputs_history.items():
            if isinstance(output, dict) and output.get("error"):
                errors_found[node_id] = output["error"]
        if errors_found:
            log_status(f"[{self.agent_id}] ERRORS_DETECTED: {errors_found}")
            return {"errors_found": True, "errors": errors_found}
        log_status(f"[{self.agent_id}] INFO: No errors detected in agent outputs.")
        return {"errors_found": False, "errors": {}}
