"""Agent that integrates multiple knowledge sources into a coherent brief."""

from llm import LLMError

from .base_agent import Agent
from .registry import register_agent
# log_status is available via base_agent.py, or if not, would need to be imported if used directly here

@register_agent("KnowledgeIntegratorAgent")
class KnowledgeIntegratorAgent(Agent):
    """Combine outputs from multiple sources into a coherent brief."""

    def execute(self, inputs: dict) -> dict:  # Type hint for dict
        """Integrate various summaries into an overall knowledge brief."""
        current_system_message = self.get_formatted_system_message()
        if current_system_message.startswith("ERROR:"):
            return {"integrated_knowledge_brief": "", "error": current_system_message}

        multi_doc_synthesis = inputs.get(
            "multi_doc_synthesis", "N/A (No multi-document synthesis provided or error upstream.)"
        )
        web_research_summary = inputs.get(
            "web_research_summary", "N/A (No web research summary provided or error upstream.)"
        )
        experimental_data_summary = inputs.get(
            "experimental_data_summary", "N/A (No experimental data provided or error upstream.)"
        )

        # Check for upstream errors and prepend to content if necessary
        if inputs.get("multi_doc_synthesis_error"):
            multi_doc_synthesis = (
                f"[Error in upstream multi-document synthesis: {inputs.get('error', 'Content unavailable')}]"
            )

        if inputs.get("web_research_summary_error"):
            web_research_summary = (
                f"[Error in upstream web research: {inputs.get('error', 'Content unavailable')}]"
            )

        if inputs.get("experimental_data_summary_error"):
            experimental_data_summary = (
                f"[Error in upstream experimental data loading: {inputs.get('error', 'Content unavailable')}]"
            )


        # Truncate inputs if they are too long
        max_input_segment_len = self.config_params.get("max_input_segment_len", 10000)
        if len(multi_doc_synthesis) > max_input_segment_len:
            multi_doc_synthesis = multi_doc_synthesis[:max_input_segment_len] + "\n[...truncated due to length...]"
        if len(web_research_summary) > max_input_segment_len:
            web_research_summary = web_research_summary[:max_input_segment_len] + "\n[...truncated due to length...]"
        # Experimental data is often shorter, but could be truncated if necessary:
        # if len(experimental_data_summary) > max_input_segment_len:
        #     experimental_data_summary = experimental_data_summary[:max_input_segment_len] + "\n[...truncated due to length...]"

        prompt = (
            f"Integrate the following information sources into a comprehensive knowledge brief as per your role:\n\n"
            f"1. Cross-Document Synthesis from multiple papers:\n---\n{multi_doc_synthesis}\n---\n\n"
            f"2. Web Research Summary:\n---\n{web_research_summary}\n---\n\n"
            f"3. Experimental Data Summary:\n---\n{experimental_data_summary}\n---\n\n"
            # The prompt from config.json already contains detailed instructions on what to include.
            # This part of the user prompt can be more concise now.
            f"Provide the integrated knowledge brief based on these sources, adhering to the specific analytical points outlined in your primary instructions (synergies, conflicts, gaps, contradictions, unanswered questions, novel links, limitations)."
        )

        temperature = float(self.config_params.get("temperature", 0.6))
        try:
            integrated_brief = self.llm.complete(
                system=current_system_message,
                prompt=prompt,
                model=self.model_name,
                temperature=temperature,
            )
        except LLMError as e:
            return {"integrated_knowledge_brief": "", "error": str(e)}
        return {"integrated_knowledge_brief": integrated_brief}
