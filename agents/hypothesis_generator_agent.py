"""Agent for generating hypotheses from an integrated knowledge brief."""

import json
from llm import LLMError
from utils import log_status
from .base_agent import Agent
from .registry import register_agent


@register_agent("HypothesisGeneratorAgent")
class HypothesisGeneratorAgent(Agent):
    """Produce research hypotheses from an integrated knowledge brief."""

    def execute(self, inputs: dict) -> dict:  # Type hint for dict
        """Generate a list of hypotheses and key opportunities."""
        num_hypotheses = self.config_params.get("num_hypotheses", 3)
        try:
            num_hypotheses = int(num_hypotheses)
            if num_hypotheses <= 0:
                num_hypotheses = 3
                log_status(
                    f"[{self.agent_id}] WARNING: Invalid 'num_hypotheses' config: {self.config_params.get('num_hypotheses')}. Defaulting to 3."
                )
        except ValueError:
            num_hypotheses = 3
            log_status(
                f"[{self.agent_id}] WARNING: Non-integer 'num_hypotheses' config: {self.config_params.get('num_hypotheses')}. Defaulting to 3."
            )

        format_args = {"num_hypotheses": num_hypotheses}
        current_system_message = self.get_formatted_system_message(format_kwargs=format_args)
        if current_system_message.startswith("ERROR:"):
            return {
                "hypotheses_output_blob": "",
                "hypotheses_list": [],
                "key_opportunities": "",
                "error": current_system_message,
            }

        integrated_brief = inputs.get("integrated_knowledge_brief")
        if inputs.get("integrated_knowledge_brief_error") or not integrated_brief:
            error_msg = (
                "Invalid or missing integrated knowledge brief for hypothesis generation. "
                f"Upstream error: {inputs.get('error', integrated_brief)}"
            )
            log_status(f"[{self.agent_id}] INPUT_ERROR: {error_msg}")
            return {
                "hypotheses_output_blob": "",
                "hypotheses_list": [],
                "key_opportunities": "",
                "error": error_msg,
            }

        max_brief_len = self.config_params.get("max_brief_len", 15000)
        if len(integrated_brief) > max_brief_len:
            log_status(
                f"[{self.agent_id}] INFO: Truncating integrated_knowledge_brief from {len(integrated_brief)} to {max_brief_len} for hypothesis generation."
            )
            integrated_brief = integrated_brief[:max_brief_len]

        user_prompt = (
            f"Based on the following 'Integrated Knowledge Brief':\n\n---\n{integrated_brief}\n---\n\n"
            f"Please provide your analysis, key research opportunities, and propose exactly {num_hypotheses} hypotheses "
            "strictly in the specified JSON format."
        )
        log_status(f"[{self.agent_id}] Requesting {num_hypotheses} hypotheses from LLM.")
        temperature = float(self.config_params.get("temperature", 0.6))
        try:
            llm_response_str = self.llm.complete(
                system=current_system_message,
                prompt=user_prompt,
                model=self.model_name,
                temperature=temperature,
            )
        except LLMError as e:
            return {
                "hypotheses_output_blob": "",
                "hypotheses_list": [],
                "key_opportunities": "",
                "error": f"LLM call failed: {e}",
            }

        return self._parse_llm_output(llm_response_str)

    def _parse_llm_output(self, llm_response_str: str) -> dict:
        """Parse the LLM response JSON into structured hypotheses."""
        try:
            cleaned = llm_response_str.strip()
            if cleaned.startswith("```json"):
                cleaned = cleaned[len("```json"):].strip()
            if cleaned.endswith("```"):
                cleaned = cleaned[:-len("```")].strip()

            parsed_output = json.loads(cleaned)
            key_opportunities = parsed_output.get("key_opportunities", "")
            hypotheses_data = parsed_output.get("hypotheses", [])
            if not isinstance(key_opportunities, str):
                key_opportunities = str(key_opportunities)

            valid_hypotheses_list = []
            if isinstance(hypotheses_data, list):
                for idx, hypo_item in enumerate(hypotheses_data):
                    if (
                        isinstance(hypo_item, dict)
                        and isinstance(hypo_item.get("hypothesis"), str)
                        and isinstance(hypo_item.get("justification"), str)
                    ):
                        valid_hypotheses_list.append(
                            {
                                "hypothesis": hypo_item["hypothesis"].strip(),
                                "justification": hypo_item["justification"].strip(),
                            }
                        )
                    else:
                        log_status(
                            f"[{self.agent_id}] WARNING: Invalid hypothesis item structure at index {idx}. Item: {str(hypo_item)[:100]}. Expected dict with 'hypothesis' and 'justification' strings."
                        )
            else:
                log_status(
                    f"[{self.agent_id}] WARNING: 'hypotheses' field is not a list or missing. LLM Output: {cleaned[:500]}"
                )

            log_status(
                f"[{self.agent_id}] Successfully parsed hypotheses. Opportunities: '{key_opportunities[:50]}...', Valid hypotheses count: {len(valid_hypotheses_list)}"
            )
            return {
                "hypotheses_output_blob": llm_response_str,
                "hypotheses_list": valid_hypotheses_list,
                "key_opportunities": key_opportunities,
            }
        except json.JSONDecodeError as e:
            log_status(
                f"[{self.agent_id}] HYPOTHESIS_PARSE_ERROR: Failed to parse JSON from LLM response. Error: {e}. Response (first 500 chars): {llm_response_str[:500]}..."
            )
            return {
                "hypotheses_output_blob": llm_response_str,
                "hypotheses_list": [],
                "key_opportunities": "",
                "error": f"Failed to parse JSON from LLM: {e}. Response: {llm_response_str[:200]}",
            }
        except (ValueError, TypeError) as e:
            log_status(
                f"[{self.agent_id}] HYPOTHESIS_UNEXPECTED_PARSE_ERROR: An error occurred while parsing LLM output: {e}. Response (first 500 chars): {llm_response_str[:500]}..."
            )
            return {
                "hypotheses_output_blob": llm_response_str,
                "hypotheses_list": [],
                "key_opportunities": "",
                "error": f"Error parsing LLM output: {e}",
            }
