from .base_agent import Agent
from utils import log_status

class ObserverAgent(Agent):
    """Agent that reviews outputs from previous agents and reports any errors."""

    def execute(self, inputs: dict) -> dict:
        outputs_history = inputs.get("outputs_history")
        if not isinstance(outputs_history, dict):
            log_status(f"[{self.agent_id}] INPUT_ERROR: 'outputs_history' missing or not a dict.")
            return {"observer_report": "", "error": "Invalid or missing outputs_history."}

        report_lines = [f"Observer review of {len(outputs_history)} outputs:"]
        errors = []
        for node_id, output in outputs_history.items():
            if isinstance(output, dict) and output.get("error"):
                errors.append(f"{node_id}: {output['error']}")
            elif isinstance(output, str) and output.startswith("Error:"):
                errors.append(f"{node_id}: {output}")
        if errors:
            report_lines.append("Errors detected:")
            report_lines.extend(f"- {e}" for e in errors)
        else:
            report_lines.append("No errors detected.")
        report = "\n".join(report_lines)
        log_status(f"[{self.agent_id}] OBSERVER_REPORT:\n{report}")
        return {"observer_report": report, "errors_found": len(errors)}
