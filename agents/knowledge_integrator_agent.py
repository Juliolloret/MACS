"""Agent that integrates multiple knowledge sources into a coherent brief."""

import json

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

        def _stringify(value):
            if isinstance(value, str):
                return value
            if value is None:
                return ""
            try:
                return json.dumps(value, ensure_ascii=False, indent=2)
            except TypeError:
                return str(value)

        max_input_segment_len = self.config_params.get("max_input_segment_len", 10000)
        sources_config = [
            (
                "multi_doc_synthesis",
                "Cross-document synthesis derived from the PDF corpus",
                "N/A (No multi-document synthesis provided or error upstream.)",
            ),
            (
                "deep_research_summary",
                "Deep-research summary grounded in the user query",
                "N/A (No deep-research synthesis provided or error upstream.)",
            ),
            (
                "web_research_summary",
                "Web research summary providing broader context",
                "N/A (No web research summary provided or error upstream.)",
            ),
            (
                "experimental_data_summary",
                "Experimental data summary",
                "N/A (No experimental data provided or error upstream.)",
            ),
        ]

        sections: list[tuple[str, str]] = []
        structured_sections: list[dict[str, str]] = []
        contributing_agents: set[str] = set()
        agent_output_details: list[dict[str, str]] = []

        def _truncate_text(text: str, limit: int) -> str:
            if len(text) > limit:
                return text[:limit] + "\n[...truncated due to length...]"
            return text

        def append_section(label_text: str, section_text: str, *, limit: int | None = None) -> None:
            effective_limit = max_input_segment_len if limit is None else limit
            normalized_text = _truncate_text(section_text or "", effective_limit)
            sections.append((label_text, normalized_text))
            structured_sections.append({"title": label_text, "content": normalized_text})

        for key, label, default_text in sources_config:
            raw_content = inputs.get(key)
            section_text = _stringify(raw_content)
            if not section_text or not section_text.strip():
                section_text = default_text
            if inputs.get(f"{key}_error"):
                err_msg = (
                    inputs.get(f"{key}_error_message")
                    or inputs.get("error")
                    or "Content unavailable"
                )
                source_name = inputs.get(f"{key}_source", "unknown upstream agent")
                section_text = (
                    f"[Error reported by {source_name} for '{key}': {err_msg}]"
                )
            source_agent = inputs.get(f"{key}_source")
            if source_agent:
                contributing_agents.add(source_agent)
                label_text = f"{label} (source: {source_agent})"
            else:
                label_text = label
            append_section(label_text, section_text)

        upstream_error_details = inputs.get("upstream_error_details") or []
        if upstream_error_details:
            formatted_error_lines: list[str] = []
            seen_pairs: set[tuple[str, str, str]] = set()
            for detail in upstream_error_details:
                source_name = detail.get("source", "unknown source")
                target_name = detail.get("target", "unknown input")
                message_text = detail.get("message", "unspecified issue")
                dedupe_key = (str(source_name), str(target_name), str(message_text))
                if dedupe_key in seen_pairs:
                    continue
                seen_pairs.add(dedupe_key)
                formatted_error_lines.append(
                    f"- Source '{source_name}' -> input '{target_name}': {message_text}"
                )
            if formatted_error_lines:
                append_section(
                    "Upstream issues detected",
                    "\n".join(formatted_error_lines),
                )

        known_keys = {cfg[0] for cfg in sources_config}
        for key in sorted(inputs.keys()):
            if key in known_keys or key.endswith(("_error", "_error_message", "_source")):
                continue
            source_agent = inputs.get(f"{key}_source")
            if not source_agent:
                continue
            supplemental_text = _stringify(inputs.get(key))
            if not supplemental_text or not supplemental_text.strip():
                continue
            if inputs.get(f"{key}_error"):
                err_msg = (
                    inputs.get(f"{key}_error_message")
                    or inputs.get("error")
                    or "Content unavailable"
                )
                supplemental_text = (
                    f"[Error reported by {source_agent} for '{key}': {err_msg}]"
                )
            contributing_agents.add(source_agent)
            append_section(
                f"Additional context from {source_agent} ({key})",
                supplemental_text,
            )

        all_agent_outputs = inputs.get("all_agent_outputs")
        if isinstance(all_agent_outputs, dict):
            for agent_name, agent_data in sorted(all_agent_outputs.items()):
                if agent_name == self.agent_id:
                    continue
                agent_text = _stringify(agent_data)
                if not agent_text or not agent_text.strip():
                    continue
                trimmed_text = _truncate_text(agent_text, max_input_segment_len)
                agent_output_details.append(
                    {"agent_id": agent_name, "content": trimmed_text}
                )
                contributing_agents.add(agent_name)
            if agent_output_details:
                aggregated_text = "\n\n".join(
                    f"{item['agent_id']}: {item['content']}"
                    for item in agent_output_details
                )
                append_section(
                    "Aggregated upstream agent outputs",
                    aggregated_text,
                    limit=max_input_segment_len * 2,
                )

        if not sections:
            append_section(
                "No upstream context provided",
                "N/A - Upstream agents did not supply any usable context.",
            )

        contributor_line = ""
        if contributing_agents:
            contributor_line = (
                "Participating agents: "
                + ", ".join(sorted(contributing_agents))
                + ".\n\n"
            )

        prompt_sections = []
        for idx, (label_text, section_text) in enumerate(sections, start=1):
            prompt_sections.append(
                f"{idx}. {label_text}:\n---\n{section_text}\n---\n"
            )

        prompt = (
            "Integrate the following information sources into a comprehensive knowledge brief as per your role:\n\n"
            f"{contributor_line}{''.join(prompt_sections)}"
            "Provide the integrated knowledge brief based on these sources, adhering to the specific analytical points outlined in your primary instructions (synergies, conflicts, gaps, contradictions, unanswered questions, novel links, limitations)."
        )

        temperature = float(self.config_params.get("temperature", 0.6))
        reasoning_effort = self.config_params.get("reasoning_effort")
        verbosity = self.config_params.get("verbosity")
        extra_params = {}
        if reasoning_effort:
            extra_params["reasoning"] = {"effort": reasoning_effort}
        if verbosity:
            extra_params["text"] = {"verbosity": verbosity}
        try:
            integrated_brief = self.llm.complete(
                system=current_system_message,
                prompt=prompt,
                model=self.model_name,
                temperature=temperature,
                extra=extra_params,
            )
        except LLMError as e:
            return {"integrated_knowledge_brief": "", "error": str(e)}
        return {
            "integrated_knowledge_brief": integrated_brief,
            "knowledge_sections": structured_sections,
            "contributing_agents": sorted(contributing_agents),
            "agent_context_details": agent_output_details,
        }
