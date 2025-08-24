import asyncio
import traceback
from typing import Dict, Any, Optional, List
from .base_agent import Agent
from .registry import register_agent
from .sdk_models import WebSearchPlan, ReportData  # Moved models
# Utilities and SDK components are now imported from utils.py
from utils import (
    APP_CONFIG, get_model_name, log_status # Utilities
)

# Handle optional import of the openai-agents SDK
try:
    from openai_agents import Agent as SDSAgent, Runner, WebSearchTool, ModelSettings
    SDK_AVAILABLE = True
except ImportError:
    SDK_AVAILABLE = False
    # Define placeholder classes to avoid NameError during class definition
    class SDSAgent: pass
    class Runner: pass
    class WebSearchTool: pass
    class ModelSettings: pass

@register_agent("WebResearcherAgent")
class WebResearcherAgent(Agent):
    def execute(self, inputs: dict) -> dict:
        if not SDK_AVAILABLE:
            sdk_unavailable_msg = (
                "OpenAI Agents SDK is not available or failed to load. Cannot perform SDK-based web research.")
            log_status(f"[{self.agent_id}] ERROR: {sdk_unavailable_msg}")
            return {"web_summary": "", "error": sdk_unavailable_msg + " Please install 'openai-agents'."}

        current_system_message = self.get_formatted_system_message()
        if current_system_message.startswith("ERROR:"):
            return {"web_summary": "", "error": current_system_message}

        query = inputs.get("cross_document_understanding")
        if inputs.get("cross_document_understanding_error") or not query or (
                isinstance(query, str) and query.startswith("Error:")):
            error_msg = (
                f"Invalid or missing 'cross_document_understanding' input for web research. Upstream error: {inputs.get('error', query)}")
            log_status(f"[{self.agent_id}] INPUT_ERROR: {error_msg}")
            return {"web_summary": "", "error": error_msg}
        log_status(f"[{self.agent_id}] Initiating SDK-based web research for query: '{query[:150]}...'")
        try:
            report_data_dict = asyncio.run(self._perform_sdk_research(query))
            if "error" in report_data_dict and report_data_dict["error"]:
                log_status(f"[{self.agent_id}] SDK_FLOW_ERROR: {report_data_dict['error']}")
                return {"web_summary": "", "error": report_data_dict['error']}
            web_summary_output = report_data_dict.get("markdown_report", "")
            if not web_summary_output:
                web_summary_output = report_data_dict.get(
                    "short_summary",
                    "Research process completed, but no detailed report content was generated.",
                )
            follow_ups = report_data_dict.get("follow_up_questions", [])
            if follow_ups:
                log_status(
                    f"[{self.agent_id}] SDK: Suggested follow-up questions: {follow_ups}"
                )
            log_status(
                f"[{self.agent_id}] SDK-based web research completed successfully. Output summary length: {len(web_summary_output)}"
            )
            return {"web_summary": web_summary_output}
        except Exception as e:
            detailed_error = traceback.format_exc()
            log_status(
                f"[{self.agent_id}] CRITICAL_EXECUTION_ERROR: Failed to run SDK research flow: {e}\n{detailed_error}")
            return {"web_summary": "", "error": f"Critical failure in SDK research execution: {e}. See logs."}

    async def _perform_sdk_research(self, query: str) -> Dict[str, Any]:
        try:
            # sdk_api_key = APP_CONFIG.get("system_variables", {}).get("openai_api_key")
            # Handled by global set_default_openai_key in main run function based on current design

            sdk_models_config = self.app_config.get("system_variables", {}).get("models", {})
            sdk_planner_model_name = get_model_name(self.app_config, sdk_models_config.get("sdk_planner_model", "gpt-4o"))
            sdk_search_model_name = get_model_name(self.app_config, sdk_models_config.get("sdk_search_model", "gpt-4o"))
            sdk_writer_model_name = get_model_name(self.app_config, sdk_models_config.get("sdk_writer_model", "gpt-4-turbo"))

            sdk_prompts_config = self.app_config.get("agent_prompts", {})
            default_planner_instr = (
                "You are a meticulous research planning assistant. Based on the user's query, "
                "devise a concise and effective plan of 3-5 distinct web search queries. "
                "For each query, clearly state the search term and a brief, specific reason "
                "explaining why this search is crucial for addressing the main query. "
                "Ensure queries are targeted and diverse enough to cover key aspects. "
                "Output this plan in the specified structured format."
            )
            default_searcher_instr = (
                "You are an efficient web search execution assistant. You will receive a specific search query "
                "and its corresponding reason. Execute this query using the provided `WebSearchTool`. "
                "Concisely summarize the most relevant findings from the search results in 2-3 focused paragraphs (under 300 words). "
                "Prioritize factual information and key insights. Avoid conversational fluff. "
                "Your output should be only this summary, ready for a synthesizing agent."
            )
            default_writer_instr = (
                "You are a senior research analyst and expert report writer. You have been provided with "
                "an original research query and a collection of summarized search snippets from various web searches. "
                "Your task is to synthesize this information into a comprehensive, well-structured, and insightful "
                "markdown report. The report should be detailed (aim for 500-1000 words), "
                "addressing the original query thoroughly. Include a very brief (2-3 sentences) executive summary at the beginning "
                "of your output. Also, suggest 2-3 pertinent follow-up questions or areas for further investigation based on your findings. "
                "Structure your entire output according to the specified format."
            )
            planner_instructions = sdk_prompts_config.get("sdk_planner_sm", default_planner_instr)
            searcher_instructions = sdk_prompts_config.get("sdk_searcher_sm", default_searcher_instr)
            writer_instructions = sdk_prompts_config.get("sdk_writer_sm", default_writer_instr)
            log_status(
                f"[{self.agent_id}] SDK: Initializing SDK agents - Planner: {sdk_planner_model_name}, Searcher: {sdk_search_model_name}, Writer: {sdk_writer_model_name}")

            sdk_planner = SDSAgent(name="SDKResearchPlanner", instructions=planner_instructions,
                                   model=sdk_planner_model_name, output_type=WebSearchPlan)
            sdk_searcher = SDSAgent(name="SDKWebSearcher", instructions=searcher_instructions,
                                    model=sdk_search_model_name, tools=[WebSearchTool()],
                                    model_settings=ModelSettings(tool_choice="required"), output_type=str)
            sdk_writer = SDSAgent(name="SDKReportWriter", instructions=writer_instructions, model=sdk_writer_model_name,
                                  output_type=ReportData)

            log_status(f"[{self.agent_id}] SDK: Step 1/3 - Planning searches for query: '{query[:70]}...'")
            planner_run_result = await Runner.run(sdk_planner, input=f"User Research Query: {query}")

            if not planner_run_result or not hasattr(planner_run_result,
                                                     'final_output_as') or not planner_run_result.is_done:
                log_status(
                    f"[{self.agent_id}] SDK_ERROR: Planner agent did not complete successfully or provide expected output structure.")
                return {"error": "SDK Planner agent failed to produce a valid plan."}
            search_plan: Optional[WebSearchPlan] = None
            try:
                search_plan = planner_run_result.final_output_as(WebSearchPlan)
            except Exception as e:
                log_status(
                    f"[{self.agent_id}] SDK_ERROR: Failed to parse planner output into WebSearchPlan: {e}. Output was: {planner_run_result.final_output}")
                return {"error": f"SDK Planner output parsing error: {e}"}
            if not search_plan or not search_plan.searches:
                log_status(
                    f"[{self.agent_id}] SDK_WARNING: Planner agent returned an empty or invalid search plan. Proceeding without web searches.")
                search_plan = WebSearchPlan(searches=[])
            log_status(f"[{self.agent_id}] SDK: Plan created with {len(search_plan.searches)} search queries.")

            search_tasks = []
            if search_plan.searches:
                log_status(f"[{self.agent_id}] SDK: Step 2/3 - Preparing {len(search_plan.searches)} search tasks...")
                for i, item in enumerate(search_plan.searches):
                    log_status(f"[{self.agent_id}] SDK:   - Task {i + 1}: Query='{item.query}', Reason='{item.reason}'")
                    search_input = f"Search Term: {item.query}\nReason for this specific search: {item.reason}"
                    search_tasks.append(Runner.run(sdk_searcher, input=search_input))

            search_summaries: List[str] = []
            if search_tasks:
                log_status(f"[{self.agent_id}] SDK: Executing search tasks in parallel...")
                search_run_results = await asyncio.gather(*search_tasks, return_exceptions=True)
                for i, res_or_exc in enumerate(search_run_results):
                    search_item_query = search_plan.searches[i].query # Safe access
                    if isinstance(res_or_exc, Exception):
                        log_status(
                            f"[{self.agent_id}] SDK_SEARCH_ERROR: Search task for '{search_item_query}' failed: {res_or_exc}")
                        search_summaries.append(
                            f"[Error: Search for '{search_item_query}' failed: {type(res_or_exc).__name__}]")
                    elif res_or_exc and hasattr(res_or_exc, 'final_output') and res_or_exc.final_output is not None:
                        summary = str(res_or_exc.final_output)
                        search_summaries.append(summary)
                        log_status(
                            f"[{self.agent_id}] SDK: Search for '{search_item_query}' completed. Summary length: {len(summary)}")
                    else:
                        log_status(
                            f"[{self.agent_id}] SDK_SEARCH_WARNING: Search task for '{search_item_query}' returned no output or unexpected structure. Result: {res_or_exc}")
                        search_summaries.append(
                            f"[Warning: No specific result or empty summary for '{search_item_query}'.]")
            else:
                log_status(f"[{self.agent_id}] SDK: No search tasks were executed based on the plan.")

            log_status(
                f"[{self.agent_id}] SDK: Step 3/3 - Synthesizing report from {len(search_summaries)} search summaries...")
            writer_input_content = f"Original Research Query: {query}\n\n"
            if search_summaries:
                writer_input_content += "Collected Search Summaries (each from a distinct search query):\n\n"
                for i, summary in enumerate(search_summaries):
                    writer_input_content += f"--- Summary {i + 1} ---\n{summary}\n\n"
            else:
                writer_input_content += "No web search summaries were available or generated to inform this report."

            writer_result_stream = Runner.run_streamed(sdk_writer, input=writer_input_content)

            # Process the stream to ensure completion
            async for event in writer_result_stream.stream_events():
                pass # We are interested in the final output

            if not writer_result_stream.is_done() or not hasattr(writer_result_stream, 'final_output_as'):
                log_status(
                    f"[{self.agent_id}] SDK_ERROR: Writer agent stream did not complete or lacks final output method.")
                return {"error": "SDK Writer agent stream processing failed."}
            final_report_data: Optional[ReportData] = None
            try:
                final_report_data = writer_result_stream.final_output_as(ReportData)
            except Exception as e:
                log_status(
                    f"[{self.agent_id}] SDK_ERROR: Failed to parse writer output into ReportData: {e}. Raw output: {writer_result_stream.final_output}")
                return {"error": f"SDK Writer output parsing error: {e}"}

            if not final_report_data: # Check if final_report_data is None after parsing attempt
                log_status(f"[{self.agent_id}] SDK_ERROR: Writer agent failed to produce a valid ReportData object.")
                return {"error": "SDK Writer agent did not produce a usable report object."}

            log_status(
                f"[{self.agent_id}] SDK: Report generation complete. Short summary: '{final_report_data.short_summary[:100]}...'")
            return {"short_summary": final_report_data.short_summary,
                    "markdown_report": final_report_data.markdown_report,
                    "follow_up_questions": final_report_data.follow_up_questions, "error": None}
        except ImportError as e:
            log_status(f"[{self.agent_id}] SDK_RUNTIME_IMPORT_ERROR in _perform_sdk_research: {e}.")
            return {"error": f"SDK Runtime Import Error: {e}. Web research functionality is unavailable."}
        except Exception as e:
            detailed_error = traceback.format_exc()
            log_status(f"[{self.agent_id}] SDK_CRITICAL_INTERNAL_ERROR in _perform_sdk_research: {e}\n{detailed_error}")
            return {"error": f"Critical internal failure during SDK research process: {e}. Check logs."}
