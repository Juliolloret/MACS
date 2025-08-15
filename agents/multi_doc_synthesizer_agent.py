import os
from typing import Dict, List
from .base_agent import Agent
from .registry import register_agent
from utils import call_openai_api, log_status

@register_agent("MultiDocSynthesizerAgent")
class MultiDocSynthesizerAgent(Agent):
    def execute(self, inputs: dict) -> dict: # Type hint for dict
        current_system_message = self.get_formatted_system_message()
        if current_system_message.startswith("ERROR:"):
            return {"multi_doc_synthesis_output": "", "error": current_system_message}
        summaries_list = inputs.get("all_pdf_summaries")
        if not summaries_list or not isinstance(summaries_list, list):
            if inputs.get("all_pdf_summaries_error"):
                error_msg = f"Upstream error providing PDF summaries: {inputs.get('error', 'Unknown upstream error')}"
                log_status(f"[{self.agent_id}] INPUT_ERROR: {error_msg}")
                return {"multi_doc_synthesis_output": "", "error": error_msg}
            log_status(
                f"[{self.agent_id}] INPUT_ERROR: No PDF summaries provided or input is not a list. Received: {type(summaries_list)}")
            return {"multi_doc_synthesis_output": "", "error": "No PDF summaries provided or input is not a list."}

        valid_summaries = [s for s in summaries_list if isinstance(s, dict) and s.get("summary") and not s.get("error")]
        if not valid_summaries:
            log_status(
                f"[{self.agent_id}] INPUT_ERROR: No valid (non-error) PDF summaries available for synthesis out of {len(summaries_list)} received.")
            return {"multi_doc_synthesis_output": "", "error": "No valid PDF summaries available for synthesis."}

        formatted_summaries = []
        for i, item in enumerate(valid_summaries):
            pdf_name = os.path.basename(item.get("original_pdf_path", f"Document {i + 1}"))
            formatted_summaries.append(f"Summary from '{pdf_name}':\n{item['summary']}\n---")

        combined_summaries_text = "\n\n".join(formatted_summaries)
        max_combined_len = 30000 # Consider making this configurable
        if len(combined_summaries_text) > max_combined_len:
            log_status(
                f"[{self.agent_id}] INFO: Truncating combined summaries from {len(combined_summaries_text)} to {max_combined_len} chars for synthesis.")
            combined_summaries_text = combined_summaries_text[:max_combined_len]

        prompt = f"Synthesize the following collection of summaries from multiple academic documents:\n\n{combined_summaries_text}\n\nProvide a coherent 'cross-document understanding' as per your role description."
        synthesis_output = call_openai_api(prompt, current_system_message, self.agent_id, model_name=self.model_name,
                                           temperature=0.6)
        if synthesis_output.startswith("Error:"): return {"multi_doc_synthesis_output": "", "error": synthesis_output}
        return {"multi_doc_synthesis_output": synthesis_output}
