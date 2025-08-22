import json
from .base_agent import Agent
from .registry import register_agent
from utils import log_status

@register_agent("HypothesisGeneratorAgent")
class HypothesisGeneratorAgent(Agent):
    def execute(self, inputs: dict) -> dict: # Type hint for dict
        num_hypotheses_to_generate = self.config_params.get("num_hypotheses", 3)
        try:
            num_hypotheses_to_generate = int(num_hypotheses_to_generate)
            if num_hypotheses_to_generate <= 0:
                num_hypotheses_to_generate = 3 # Default to 3 if invalid
                log_status(
                    f"[{self.agent_id}] WARNING: Invalid 'num_hypotheses' config: {self.config_params.get('num_hypotheses')}. Defaulting to 3.")
        except ValueError:
            num_hypotheses_to_generate = 3 # Default to 3 if not an int
            log_status(
                f"[{self.agent_id}] WARNING: Non-integer 'num_hypotheses' config: {self.config_params.get('num_hypotheses')}. Defaulting to 3.")

        format_args = {"num_hypotheses": num_hypotheses_to_generate}
        current_system_message = self.get_formatted_system_message(format_kwargs=format_args)

        if current_system_message.startswith("ERROR:"):
            return {"hypotheses_output_blob": "", "hypotheses_list": [], "key_opportunities": "",
                    "error": current_system_message}

        integrated_knowledge_brief = inputs.get("integrated_knowledge_brief")
        if inputs.get("integrated_knowledge_brief_error") or not integrated_knowledge_brief or \
                (isinstance(integrated_knowledge_brief, str) and integrated_knowledge_brief.startswith("Error:")):
            error_msg = f"Invalid or missing integrated knowledge brief for hypothesis generation. Upstream error: {inputs.get('error', integrated_knowledge_brief)}"
            log_status(f"[{self.agent_id}] INPUT_ERROR: {error_msg}")
            return {"hypotheses_output_blob": "", "hypotheses_list": [], "key_opportunities": "", "error": error_msg}

        max_brief_len = 15000 # Consider making this configurable
        if len(integrated_knowledge_brief) > max_brief_len:
            log_status(
                f"[{self.agent_id}] INFO: Truncating integrated_knowledge_brief from {len(integrated_knowledge_brief)} to {max_brief_len} for hypothesis generation.")
            integrated_knowledge_brief = integrated_knowledge_brief[:max_brief_len]

        user_prompt = (
            f"Based on the following 'Integrated Knowledge Brief':\n\n---\n{integrated_knowledge_brief}\n---\n\n"
            f"Please provide your analysis, key research opportunities, and propose exactly {num_hypotheses_to_generate} hypotheses "
            "strictly in the specified JSON format."
        )
        log_status(f"[{self.agent_id}] Requesting {num_hypotheses_to_generate} hypotheses from LLM.")
        llm_response_str = self.llm.complete(
            system=current_system_message,
            prompt=user_prompt,
            model=self.model_name,
        )

        if llm_response_str.startswith("Error:"):
            return {"hypotheses_output_blob": llm_response_str, "hypotheses_list": [], "key_opportunities": "",
                    "error": f"LLM call failed: {llm_response_str}"}

        try:
            # Attempt to clean up markdown code block fences if present
            cleaned_llm_response_str = llm_response_str
            if cleaned_llm_response_str.strip().startswith("```json"):
                cleaned_llm_response_str = cleaned_llm_response_str.strip()[len("```json"):].strip()
            if cleaned_llm_response_str.strip().endswith("```"):
                cleaned_llm_response_str = cleaned_llm_response_str.strip()[:-len("```")].strip()

            parsed_output = json.loads(cleaned_llm_response_str)
            key_opportunities = parsed_output.get("key_opportunities", "")
            hypotheses_data = parsed_output.get("hypotheses", []) # This should be a list of objects now

            if not isinstance(key_opportunities, str):
                key_opportunities = str(key_opportunities) # Ensure string

            valid_hypotheses_list = []
            if isinstance(hypotheses_data, list):
                for idx, hypo_item in enumerate(hypotheses_data):
                    if isinstance(hypo_item, dict) and \
                       isinstance(hypo_item.get('hypothesis'), str) and \
                       isinstance(hypo_item.get('justification'), str):
                        valid_hypotheses_list.append({
                            "hypothesis": hypo_item['hypothesis'].strip(),
                            "justification": hypo_item['justification'].strip()
                        })
                    else:
                        log_status(f"[{self.agent_id}] WARNING: Invalid hypothesis item structure at index {idx}. Item: {str(hypo_item)[:100]}. Expected dict with 'hypothesis' and 'justification' strings.")
            else:
                log_status(f"[{self.agent_id}] WARNING: 'hypotheses' field is not a list or missing. LLM Output: {cleaned_llm_response_str[:500]}")
                # valid_hypotheses_list remains empty

            log_status(
                f"[{self.agent_id}] Successfully parsed hypotheses. Opportunities: '{key_opportunities[:50]}...', Valid hypotheses count: {len(valid_hypotheses_list)}")
            return {
                "hypotheses_output_blob": llm_response_str, # Original response for debugging
                "hypotheses_list": valid_hypotheses_list,   # List of dicts
                "key_opportunities": key_opportunities
            }
        except json.JSONDecodeError as e:
            log_status(
                f"[{self.agent_id}] HYPOTHESIS_PARSE_ERROR: Failed to parse JSON from LLM response. Error: {e}. Response (first 500 chars): {llm_response_str[:500]}...")
            return {"hypotheses_output_blob": llm_response_str, "hypotheses_list": [], "key_opportunities": "",
                    "error": f"Failed to parse JSON from LLM: {e}. Response: {llm_response_str[:200]}"}
        except Exception as e: # Catch any other unexpected error during parsing
            log_status(
                f"[{self.agent_id}] HYPOTHESIS_UNEXPECTED_PARSE_ERROR: An unexpected error occurred while parsing LLM output: {e}. Response (first 500 chars): {llm_response_str[:500]}...")
            return {"hypotheses_output_blob": llm_response_str, "hypotheses_list": [], "key_opportunities": "",
                    "error": f"Unexpected error parsing LLM output: {e}"}
