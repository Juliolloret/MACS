"""Agent for designing experiments based on hypotheses."""

from utils import log_status
from llm import LLMError
from .base_agent import Agent
from .registry import register_agent

@register_agent("ExperimentDesignerAgent")
class ExperimentDesignerAgent(Agent):
    """Generate experimental protocols for a list of hypotheses."""

    def execute(self, inputs: dict) -> dict:  # Type hint for dict
        """Produce experiment designs for each hypothesis in ``inputs``."""
        current_system_message = self.get_formatted_system_message()
        if current_system_message.startswith("ERROR:"):
            return {"experiment_designs_list": [], "error": current_system_message}

        hypotheses_list_input = inputs.get("hypotheses_list", []) # Expects a list of hypothesis objects
        hypotheses_list_had_upstream_error = inputs.get("hypotheses_list_error", False) # Check for upstream error flag

        if hypotheses_list_had_upstream_error:
            log_status(f"[{self.agent_id}] INPUT_ERROR: Upstream error indicated for hypotheses_list. Cannot design experiments.")
            return {"experiment_designs_list": [], "error": "Upstream error in hypotheses list provided."}

        if not isinstance(hypotheses_list_input, list) or not hypotheses_list_input:
            log_status(
                f"[{self.agent_id}] INFO: No hypotheses provided or invalid format for experiment design. Input type: {type(hypotheses_list_input)}")
            return {"experiment_designs_list": [], "info": "No valid hypotheses provided to design experiments for."}

        all_designs = []
        for i, hypo_obj in enumerate(hypotheses_list_input):
            # Validate the structure of the hypothesis object
            if not isinstance(hypo_obj, dict) or \
               not isinstance(hypo_obj.get('hypothesis'), str) or not hypo_obj['hypothesis'].strip() or \
               not isinstance(hypo_obj.get('justification'), str): # Checking justification presence and type
                log_status(f"[{self.agent_id}] WARNING: Skipping invalid hypothesis object at index {i}: '{str(hypo_obj)[:150]}'. Expected dict with non-empty 'hypothesis' (str) and 'justification' (str).")
                all_designs.append({
                    "experiment_design": "",
                    "hypothesis_processed": str(hypo_obj), # Log the problematic object
                    "error": "Invalid hypothesis object structure, missing/empty hypothesis string, or missing justification."
                })
                continue

            hypo_str = hypo_obj['hypothesis']
            # Log justification for context if needed, but it's not directly in the prompt to LLM here.
            log_status(f"[{self.agent_id}] Designing experiment for hypothesis {i + 1}: '{hypo_str[:100]}...' (Justification: '{hypo_obj['justification'][:50]}...')")

            prompt = (
                "Design a detailed, feasible, and rigorous experimental protocol for the following hypothesis:\n\n"
                f"Hypothesis: \"{hypo_str}\"\n\nAs per your role, include sections like Objective, Methodology & Apparatus, "
                "Step-by-step Procedure, Variables & Controls, Data Collection & Analysis, Expected Outcomes & Success Criteria, "
                "Potential Challenges & Mitigation, and Ethical Considerations (if applicable)."
            )

            design_output_single = {"hypothesis_processed": hypo_str}
            try:
                design = self.llm.complete(
                    system=current_system_message,
                    prompt=prompt,
                    model=self.model_name,
                )
                design_output_single["experiment_design"] = design
            except LLMError as e:
                design_output_single["experiment_design"] = ""
                design_output_single["error"] = str(e)
            all_designs.append(design_output_single)

        log_status(f"[{self.agent_id}] Finished designing experiments for {len(hypotheses_list_input)} hypotheses.")
        return {"experiment_designs_list": all_designs}
