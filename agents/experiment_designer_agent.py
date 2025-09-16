"""Agent for designing experiments based on hypotheses."""

from utils import log_status
from llm import LLMError
from .base_agent import Agent
from .registry import register_agent

@register_agent("ExperimentDesignerAgent")
class ExperimentDesignerAgent(Agent):
    """Generate an experimental protocol for a single hypothesis."""

    def execute(self, inputs: dict) -> dict:  # Type hint for dict
        """Produce experiment designs for one or many hypotheses."""

        current_system_message = self.get_formatted_system_message()

        hypotheses_list = inputs.get("hypotheses_list")
        # When the agent receives a batch of hypotheses (e.g., during tests or
        # when invoked outside of the orchestrator loop) we return a
        # consolidated response instead of a single design dictionary. Treat
        # the presence of the list key or an associated error flag as an
        # indication that batched handling is expected, even if the actual
        # list value is missing.
        if (
            isinstance(hypotheses_list, list)
            or "hypotheses_list" in inputs
            or inputs.get("hypotheses_list_error")
        ):
            if current_system_message.startswith("ERROR:"):
                return {"experiment_designs_list": [], "error": current_system_message}
            return self._design_for_hypothesis_list(hypotheses_list, inputs, current_system_message)

        if current_system_message.startswith("ERROR:"):
            return {"experiment_design": "", "error": current_system_message}

        had_upstream_error = inputs.get("hypothesis_error", False)
        if had_upstream_error:
            log_status(f"[{self.agent_id}] INPUT_ERROR: Upstream error indicated for hypothesis. Cannot design experiment.")
            return {"experiment_design": "", "error": "Upstream error in hypothesis provided."}

        hypo_obj = inputs.get("hypothesis", {})
        return self._design_single_hypothesis(hypo_obj, current_system_message)

    def _design_for_hypothesis_list(self, hypotheses_list, inputs, current_system_message):
        """Generate experiment designs for a list of hypotheses."""

        if inputs.get("hypotheses_list_error") or inputs.get("error"):
            error_msg = inputs.get(
                "error",
                "Upstream error flagged for hypotheses list. Cannot design experiments.",
            )
            log_status(f"[{self.agent_id}] INPUT_ERROR: {error_msg}")
            return {"experiment_designs_list": [], "error": error_msg}

        if not hypotheses_list:
            log_status(f"[{self.agent_id}] INFO: No hypotheses provided for experiment design.")
            return {
                "experiment_designs_list": [],
                "error": "No hypotheses were supplied for experiment design.",
            }

        designs = []
        errors = []
        for idx, hypothesis_entry in enumerate(hypotheses_list):
            design = self._design_single_hypothesis(hypothesis_entry, current_system_message)
            designs.append(design)
            if design.get("error"):
                errors.append({"index": idx, "error": design["error"]})

        output = {"experiment_designs_list": designs}
        if errors:
            output["errors"] = errors
        return output

    def _design_single_hypothesis(self, hypo_obj, current_system_message):
        """Generate an experiment design for a single hypothesis object."""

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
