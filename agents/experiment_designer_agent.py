"""Agent for designing experiments based on hypotheses."""

from utils import log_status
from llm import LLMError
from .base_agent import Agent
from .registry import register_agent

@register_agent("ExperimentDesignerAgent")
class ExperimentDesignerAgent(Agent):
    """Generate an experimental protocol for a single hypothesis."""

    def execute(self, inputs: dict) -> dict:  # Type hint for dict
        """Produce an experiment design for a single hypothesis in ``inputs``."""
        current_system_message = self.get_formatted_system_message()
        if current_system_message.startswith("ERROR:"):
            return {"experiment_design": "", "error": current_system_message}

        hypo_obj = inputs.get("hypothesis", {})  # Expects a single hypothesis object
        had_upstream_error = inputs.get("hypothesis_error", False)

        if had_upstream_error:
            log_status(f"[{self.agent_id}] INPUT_ERROR: Upstream error indicated for hypothesis. Cannot design experiment.")
            return {"experiment_design": "", "error": "Upstream error in hypothesis provided."}

        # Validate the structure of the hypothesis object
        if not isinstance(hypo_obj, dict) or \
           not isinstance(hypo_obj.get('hypothesis'), str) or not hypo_obj['hypothesis'].strip() or \
           not isinstance(hypo_obj.get('justification'), str):
            error_msg = f"Skipping invalid hypothesis object: '{str(hypo_obj)[:150]}'. Expected dict with non-empty 'hypothesis' (str) and 'justification' (str)."
            log_status(f"[{self.agent_id}] WARNING: {error_msg}")
            return {
                "experiment_design": "",
                "hypothesis_processed": str(hypo_obj),
                "error": error_msg
            }

        hypo_str = hypo_obj['hypothesis']
        log_status(f"[{self.agent_id}] Designing experiment for hypothesis: '{hypo_str[:100]}...'")

        prompt = (
            "Design a detailed, feasible, and rigorous experimental protocol for the following hypothesis:\n\n"
            f"Hypothesis: \"{hypo_str}\"\n\nAs per your role, include sections like Objective, Methodology & Apparatus, "
            "Step-by-step Procedure, Variables & Controls, Data Collection & Analysis, Expected Outcomes & Success Criteria, "
            "Potential Challenges & Mitigation, and Ethical Considerations (if applicable)."
        )

        design_output_single = {"hypothesis_processed": hypo_str}
        try:
            reasoning_effort = self.config_params.get("reasoning_effort")
            verbosity = self.config_params.get("verbosity")
            extra_params = {}
            if reasoning_effort:
                extra_params["reasoning"] = {"effort": reasoning_effort}
            if verbosity:
                extra_params["text"] = {"verbosity": verbosity}
            design = self.llm.complete(
                system=current_system_message,
                prompt=prompt,
                model=self.model_name,
                extra=extra_params,
            )
            design_output_single["experiment_design"] = design
        except LLMError as e:
            design_output_single["experiment_design"] = ""
            design_output_single["error"] = str(e)

        return design_output_single
