import os
import sys
import time
import shutil
import json
from collections import defaultdict, deque
import asyncio
import traceback
from typing import List, Dict, Any, Optional

# Pydantic for structured data, used by the SDK agents
from pydantic import BaseModel, Field

# --- SDK Imports & Availability Check ---
SDK_AVAILABLE = False
SDSAgent, Runner, WebSearchTool, ModelSettings = None, None, None, None
_sdk_import_error = None

try:
    from agents import (
        Agent as SDSAgent_actual,
        Runner as Runner_actual,
        WebSearchTool as WebSearchTool_actual,
        ModelSettings as ModelSettings_actual,
        set_default_openai_key  # Import the function to set the key for the SDK
    )

    SDSAgent, Runner, WebSearchTool, ModelSettings = (
        SDSAgent_actual, Runner_actual, WebSearchTool_actual, ModelSettings_actual
    )
    SDK_AVAILABLE = True
except ImportError as e:
    _sdk_import_error = e
    # Fallback if set_default_openai_key is not directly under agents in some versions
    if 'set_default_openai_key' not in globals():
        try:
            from agents.config import set_default_openai_key  # Common alternative path
        except ImportError:
            pass  # Will be handled by SDK_AVAILABLE check
    pass
# --- End SDK Imports & Availability Check ---


# --- Library Import Placeholders ---
OpenAI = None
PyPDF2 = None
REPORTLAB_AVAILABLE = False
canvas = None
letter = None
inch = None
openai_errors = None

# --- Global Configuration Variable ---
APP_CONFIG = {}
# --- Global Status Callback ---
STATUS_CALLBACK = print
# --- Script's Directory ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def set_status_callback(callback_func):
    global STATUS_CALLBACK
    STATUS_CALLBACK = callback_func


def log_status(message):
    if callable(STATUS_CALLBACK):
        STATUS_CALLBACK(message)
    else:
        print(message)


if SDK_AVAILABLE:
    log_status("INFO: openai-agents SDK loaded successfully. SDK-based WebResearcherAgent capabilities enabled.")
else:
    error_message_suffix = f" (Error: {_sdk_import_error})" if _sdk_import_error else ""
    log_status(
        f"WARNING: openai-agents SDK not found or failed to import{error_message_suffix}. "
        "WebResearcherAgent with SDK integration will be disabled. "
        "Please ensure 'openai-agents' package is installed in the correct environment."
    )


# --- Utility Functions ---
def load_app_config(config_path="config.json"):
    global APP_CONFIG, OpenAI, PyPDF2, REPORTLAB_AVAILABLE, canvas, letter, inch, openai_errors
    OpenAI = None
    PyPDF2 = None
    REPORTLAB_AVAILABLE = False
    canvas, letter, inch = None, None, None
    openai_errors = None
    resolved_config_path = config_path
    if not os.path.isabs(config_path):
        resolved_config_path = os.path.join(SCRIPT_DIR, config_path)
    log_status(f"[AppConfig] Attempting to load configuration from resolved path: '{resolved_config_path}'")
    try:
        with open(resolved_config_path, 'r', encoding='utf-8') as f:
            APP_CONFIG = json.load(f)
        log_status(f"[AppConfig] Successfully loaded configuration from '{resolved_config_path}'.")
        try:
            from openai import OpenAI as OpenAI_lib, APIConnectionError, APITimeoutError, RateLimitError, \
                AuthenticationError, BadRequestError
            OpenAI = OpenAI_lib
            openai_errors = {"APIConnectionError": APIConnectionError, "APITimeoutError": APITimeoutError,
                             "RateLimitError": RateLimitError, "AuthenticationError": AuthenticationError,
                             "BadRequestError": BadRequestError}
            log_status("[AppConfig] OpenAI library loaded.")
        except ImportError:
            log_status("[AppConfig] WARNING: OpenAI library not found (pip install openai).")
        try:
            import PyPDF2 as PyPDF2_lib
            PyPDF2 = PyPDF2_lib
            log_status("[AppConfig] PyPDF2 library loaded.")
        except ImportError:
            log_status("[AppConfig] WARNING: PyPDF2 library not found (pip install PyPDF2).")
        try:
            from reportlab.pdfgen import canvas as rl_canvas
            from reportlab.lib.pagesizes import letter as rl_letter
            from reportlab.lib.units import inch as rl_inch
            canvas, letter, inch = rl_canvas, rl_letter, rl_inch
            REPORTLAB_AVAILABLE = True
            log_status("[AppConfig] reportlab library loaded.")
        except ImportError:
            log_status("[AppConfig] WARNING: reportlab library not found. PDF output features unavailable.")
        return True
    except FileNotFoundError:
        log_status(f"[AppConfig] ERROR: Configuration file '{resolved_config_path}' not found.")
    except json.JSONDecodeError as e:
        log_status(f"[AppConfig] ERROR: Could not decode JSON from '{resolved_config_path}': {e}.")
    except Exception as e:
        log_status(
            f"[AppConfig] ERROR: An unexpected error occurred while loading config '{resolved_config_path}': {e}.")
    APP_CONFIG = {}
    return False


def get_model_name(model_key=None):
    if not APP_CONFIG: return "gpt-4o"
    models_config = APP_CONFIG.get("system_variables", {}).get("models", {})
    if model_key and model_key in models_config: return models_config[model_key]
    return APP_CONFIG.get("system_variables", {}).get("default_llm_model", "gpt-4o")


def get_prompt_text(prompt_key):
    if prompt_key is None: return ""
    if not APP_CONFIG: return f"ERROR: Config not loaded, prompt key '{prompt_key}' unavailable."
    prompts_config = APP_CONFIG.get("agent_prompts", {})
    if prompt_key not in prompts_config:
        log_status(f"[AppConfig] ERROR: Prompt key '{prompt_key}' not found in agent_prompts.")
        return f"ERROR: Prompt key '{prompt_key}' not found."
    prompt_text = prompts_config.get(prompt_key)
    if prompt_text is None:
        log_status(f"[AppConfig] WARNING: Prompt key '{prompt_key}' has null value in config. Returning empty string.")
        return ""
    return prompt_text


def call_openai_api(prompt, system_message="You are a helpful assistant.", agent_name="LLM", model_name=None,
                    temperature=0.5):
    chosen_model = model_name if model_name else get_model_name()
    effective_system_message = system_message if system_message else "You are a helpful assistant."
    if isinstance(system_message, str) and system_message.startswith("ERROR:"):
        log_status(f"[{agent_name}] LLM_CALL_ERROR: Invalid system message provided: {system_message}")
        return f"Error: Invalid system message for agent {agent_name} due to: {system_message}"
    current_temperature = temperature
    if chosen_model and "o4-mini" in chosen_model:
        log_status(
            f"[{agent_name}] INFO: Model '{chosen_model}' detected. Adjusting temperature to 1.0 (default) as it may not support other values.")
        current_temperature = 1.0
    prompt_display_snippet = prompt[:150].replace('\n', ' ')
    log_status(
        f"[{agent_name}] LLM_CALL_START: Model='{chosen_model}', Temp='{current_temperature}', SystemMessage='{effective_system_message[:70]}...', Prompt(start): '{prompt_display_snippet}...'")
    if not OpenAI: return f"Error: OpenAI library not available for model {chosen_model}."
    if not APP_CONFIG: return f"Error: Application configuration not loaded for model {chosen_model}."
    api_key_to_use = APP_CONFIG.get("system_variables", {}).get("openai_api_key")
    api_timeout_seconds = APP_CONFIG.get("system_variables", {}).get("openai_api_timeout_seconds", 120)
    if not api_key_to_use or api_key_to_use in ["YOUR_OPENAI_API_KEY_NOT_IN_CONFIG", "YOUR_ACTUAL_OPENAI_API_KEY",
                                                "KEY"]:
        return f"Error: OpenAI API key not configured for model {chosen_model}."
    try:
        client = OpenAI(api_key=api_key_to_use, timeout=api_timeout_seconds)
        api_call_params = {"model": chosen_model, "messages": [{"role": "system", "content": effective_system_message},
                                                               {"role": "user", "content": prompt}],
                           "temperature": current_temperature}
        response = client.chat.completions.create(**api_call_params)
        if not response.choices:
            log_status(f"[{agent_name}] LLM_CALL_ERROR: Model='{chosen_model}' response has no choices.")
            return f"Error: OpenAI API response had no choices for model {chosen_model}."
        first_choice = response.choices[0]
        if not first_choice.message:
            log_status(f"[{agent_name}] LLM_CALL_ERROR: Model='{chosen_model}' first choice has no message object.")
            return f"Error: OpenAI API response choice had no message object for model {chosen_model}."
        raw_content = first_choice.message.content
        if raw_content is None:
            log_status(f"[{agent_name}] LLM_CALL_SUCCESS_EMPTY_CONTENT: Model='{chosen_model}' returned None content.")
            return ""
        if not isinstance(raw_content, str):
            log_status(
                f"[{agent_name}] LLM_CALL_ERROR_UNEXPECTED_CONTENT_TYPE: Model='{chosen_model}' returned content of type {type(raw_content)}, expected string. Content: {str(raw_content)[:100]}")
            return f"Error: OpenAI API returned unexpected content type for model {chosen_model}."
        result = raw_content.strip()
        result_display_snippet = result[:150].replace('\n', ' ')
        log_status(
            f"[{agent_name}] LLM_CALL_SUCCESS: Model='{chosen_model}', Response(start): '{result_display_snippet}...'")
        return result
    except Exception as e:
        error_type_name = type(e).__name__
        if openai_errors:
            for err_name, err_class in openai_errors.items():
                if isinstance(e, err_class):
                    log_status(f"[{agent_name}] LLM_ERROR ({err_name}): API call with {chosen_model} failed: {e}")
                    error_detail = str(e)
                    if hasattr(e, 'response') and hasattr(e.response, 'text'):
                        try:
                            err_json = json.loads(e.response.text)
                            if 'error' in err_json and 'message' in err_json['error']:
                                error_detail = err_json['error']['message']
                            else:
                                error_detail = e.response.text[:500]
                        except json.JSONDecodeError:
                            error_detail = e.response.text[:500]
                        except Exception:
                            pass
                    return f"Error: OpenAI API {err_name} for {chosen_model}: {error_detail}"
        log_status(f"[{agent_name}] LLM_ERROR (General {error_type_name}): API call with {chosen_model} failed: {e}")
        return f"Error: API call with {chosen_model} failed ({error_type_name}): {e}"


class Agent:
    def __init__(self, agent_id, agent_type, config_params=None):
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.config_params = config_params if config_params else {}
        model_key_from_config = self.config_params.get("model_key")
        self.model_name = get_model_name(model_key_from_config)
        system_message_key = self.config_params.get("system_message_key")
        self.system_message = get_prompt_text(system_message_key)
        description = self.config_params.get("description", "No description provided.")
        log_status(
            f"[AgentInit] Created Agent: ID='{self.agent_id}', Type='{self.agent_type}', PrimaryModel='{self.model_name}'. SystemMsgKey='{system_message_key}'. Desc: {description}")
        if self.system_message.startswith("ERROR:"):
            log_status(
                f"[AgentInit] CRITICAL_WARNING: System message for Agent ID='{self.agent_id}' (Key: '{system_message_key}') could not be resolved: {self.system_message}")
        elif system_message_key and not self.system_message:
            log_status(
                f"[AgentInit] WARNING: System message for Agent ID='{self.agent_id}' (Key: '{system_message_key}') is empty.")

    def get_formatted_system_message(self, format_kwargs: Optional[Dict[str, Any]] = None) -> str:
        if self.system_message.startswith("ERROR:"):
            return self.system_message
        if format_kwargs:
            try:
                return self.system_message.format(**format_kwargs)
            except KeyError as e:
                log_status(
                    f"[{self.agent_id}] SYSTEM_MSG_FORMAT_ERROR: Missing key {e} for system message template. Using unformatted message. Template was: '{self.system_message[:200]}...'")
                return self.system_message
        return self.system_message

    def execute(self, inputs: dict) -> dict:
        if self.system_message.startswith("ERROR:"):
            return {
                "error": f"Agent {self.agent_id} cannot execute due to configuration error for system message (key: '{self.config_params.get('system_message_key')}'): {self.system_message}"}
        return None


class WebSearchItem(BaseModel):
    reason: str = Field(description="Reasoning for why this search is important to the query.")
    query: str = Field(description="The search term to use for the web search.")


class WebSearchPlan(BaseModel):
    searches: List[WebSearchItem] = Field(default_factory=list,
                                          description="A list of web searches to perform to best answer the query.")


class ReportData(BaseModel):
    short_summary: str = Field(description="A short 2-3 sentence summary of the findings.")
    markdown_report: str = Field(description="The final comprehensive report in markdown format.")
    follow_up_questions: List[str] = Field(default_factory=list,
                                           description="Suggested topics or questions for further research.")


class PDFLoaderAgent(Agent):
    def execute(self, inputs: dict) -> dict:
        base_pre_check_result = super().execute(inputs)
        if base_pre_check_result: return base_pre_check_result
        pdf_path = inputs.get("pdf_path")
        if not pdf_path: return {"pdf_text_content": "", "error": "PDF path not provided."}
        log_status(f"[{self.agent_id}] PDF_LOAD_START: Path='{pdf_path}'")
        if not PyPDF2: return {"pdf_text_content": "", "error": "PyPDF2 library not available."}
        if not os.path.exists(pdf_path): return {"pdf_text_content": "", "error": f"PDF file not found: {pdf_path}"}
        try:
            if os.path.getsize(pdf_path) == 0: return {"pdf_text_content": "",
                                                       "error": f"PDF file is empty: {pdf_path}"}
        except OSError as oe:
            return {"pdf_text_content": "", "error": f"Could not access file for size check: {pdf_path}, {oe}"}
        text_content = ""
        try:
            with open(pdf_path, 'rb') as pdf_file_obj:
                pdf_reader = PyPDF2.PdfReader(pdf_file_obj)
                if pdf_reader.is_encrypted:
                    if pdf_reader.decrypt('') == 0:
                        log_status(
                            f"[{self.agent_id}] PDF_LOAD_ERROR: Failed to decrypt PDF '{os.path.basename(pdf_path)}'.")
                        return {"pdf_text_content": "",
                                "error": f"Failed to decrypt PDF: {os.path.basename(pdf_path)}."}
                    log_status(f"[{self.agent_id}] PDF_LOAD_INFO: PDF '{os.path.basename(pdf_path)}' decrypted.")
                for page_obj in pdf_reader.pages: text_content += page_obj.extract_text() or ""
            if not text_content.strip(): log_status(
                f"[{self.agent_id}] PDF_LOAD_WARNING: No text extracted from '{os.path.basename(pdf_path)}'.")
            return {"pdf_text_content": text_content, "original_pdf_path": pdf_path}
        except Exception as e:
            log_status(f"[{self.agent_id}] PDF_LOAD_ERROR: PDF extraction failed for {pdf_path}: {e}")
            return {"pdf_text_content": "", "error": f"PDF extraction failed for {pdf_path}: {e}"}


class PDFSummarizerAgent(Agent):
    def execute(self, inputs: dict) -> dict:
        current_system_message = self.get_formatted_system_message()
        if current_system_message.startswith("ERROR:"):
            return {"summary": "", "error": current_system_message,
                    "original_pdf_path": inputs.get("original_pdf_path", "Unknown PDF")}
        pdf_text_content = inputs.get("pdf_text_content")
        original_pdf_path = inputs.get("original_pdf_path", "Unknown PDF")
        if inputs.get("pdf_text_content_error") or not pdf_text_content or (
                isinstance(pdf_text_content, str) and pdf_text_content.startswith("Error:")):
            error_msg = f"Invalid text content for summarization from {original_pdf_path}. Upstream error: {inputs.get('error', pdf_text_content)}"
            return {"summary": "", "error": error_msg, "original_pdf_path": original_pdf_path}
        max_len = self.config_params.get("max_input_length", 15000)
        if len(pdf_text_content) > max_len:
            log_status(
                f"[{self.agent_id}] INFO: Truncating PDF text from {len(pdf_text_content)} to {max_len} chars for summarization.")
            pdf_text_content = pdf_text_content[:max_len]
        prompt = f"Please summarize the following academic text from document '{os.path.basename(original_pdf_path)}':\n\n---\n{pdf_text_content}\n---"
        summary = call_openai_api(prompt, current_system_message, self.agent_id, model_name=self.model_name)
        if summary.startswith("Error:"): return {"summary": "", "error": summary,
                                                 "original_pdf_path": original_pdf_path}
        return {"summary": summary, "original_pdf_path": original_pdf_path}


class MultiDocSynthesizerAgent(Agent):
    def execute(self, inputs: dict) -> dict:
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
        max_combined_len = 30000
        if len(combined_summaries_text) > max_combined_len:
            log_status(
                f"[{self.agent_id}] INFO: Truncating combined summaries from {len(combined_summaries_text)} to {max_combined_len} chars for synthesis.")
            combined_summaries_text = combined_summaries_text[:max_combined_len]
        prompt = f"Synthesize the following collection of summaries from multiple academic documents:\n\n{combined_summaries_text}\n\nProvide a coherent 'cross-document understanding' as per your role description."
        synthesis_output = call_openai_api(prompt, current_system_message, self.agent_id, model_name=self.model_name,
                                           temperature=0.6)
        if synthesis_output.startswith("Error:"): return {"multi_doc_synthesis_output": "", "error": synthesis_output}
        return {"multi_doc_synthesis_output": synthesis_output}


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
            if not web_summary_output: web_summary_output = report_data_dict.get("short_summary",
                                                                                 "Research process completed, but no detailed report content was generated.")
            follow_ups = report_data_dict.get("follow_up_questions", [])
            if follow_ups: log_status(f"[{self.agent_id}] SDK: Suggested follow-up questions: {follow_ups}")
            log_status(
                f"[{self.agent_id}] SDK-based web research completed successfully. Output summary length: {len(web_summary_output)}")
            return {"web_summary": web_summary_output}
        except Exception as e:
            detailed_error = traceback.format_exc()
            log_status(
                f"[{self.agent_id}] CRITICAL_EXECUTION_ERROR: Failed to run SDK research flow: {e}\n{detailed_error}")
            return {"web_summary": "", "error": f"Critical failure in SDK research execution: {e}. See logs."}

    async def _perform_sdk_research(self, query: str) -> Dict[str, Any]:
        try:
            # Ensure API key is set for the SDK if not done globally
            # This is a good place to ensure it's set before SDK usage within this agent
            # if 'set_default_openai_key' in globals() and callable(set_default_openai_key):
            #     sdk_api_key = APP_CONFIG.get("system_variables", {}).get("openai_api_key")
            #     if sdk_api_key:
            #         set_default_openai_key(sdk_api_key)
            #         log_status(f"[{self.agent_id}] SDK: Ensured OpenAI API key is set for SDK.")
            #     else:
            #         log_status(f"[{self.agent_id}] SDK_WARNING: OpenAI API key not found in APP_CONFIG for SDK.")
            #         # Depending on SDK behavior, this might still lead to an error if env var isn't set either.

            sdk_models_config = APP_CONFIG.get("system_variables", {}).get("models", {})
            sdk_planner_model_name = get_model_name(sdk_models_config.get("sdk_planner_model", "gpt-4o"))
            sdk_search_model_name = get_model_name(sdk_models_config.get("sdk_search_model", "gpt-4o"))
            sdk_writer_model_name = get_model_name(sdk_models_config.get("sdk_writer_model", "gpt-4-turbo"))
            sdk_prompts_config = APP_CONFIG.get("agent_prompts", {})
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
                    search_item_query = search_plan.searches[i].query
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
                for i, summary in enumerate(
                    search_summaries): writer_input_content += f"--- Summary {i + 1} ---\n{summary}\n\n"
            else:
                writer_input_content += "No web search summaries were available or generated to inform this report."

            writer_result_stream = Runner.run_streamed(sdk_writer, input=writer_input_content)

            async for event in writer_result_stream.stream_events(): pass

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
            if not final_report_data:
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


class ExperimentalDataLoaderAgent(Agent):
    def execute(self, inputs: dict) -> dict:
        current_system_message = self.get_formatted_system_message()
        if current_system_message.startswith("ERROR:"):
            return {"experimental_data_summary": "", "error": current_system_message}
        data_file_path = inputs.get("experimental_data_file_path")
        if not data_file_path:
            log_status(f"[{self.agent_id}] INFO: No experimental data file path provided. Proceeding without it.")
            return {"experimental_data_summary": "N/A - No experimental data file provided."}
        resolved_data_path = data_file_path
        if not os.path.isabs(data_file_path):
            resolved_data_path = os.path.join(SCRIPT_DIR, data_file_path)
            log_status(
                f"[{self.agent_id}] INFO: Relative experimental data path '{data_file_path}' resolved to '{resolved_data_path}'.")
        if not os.path.exists(resolved_data_path):
            log_status(f"[{self.agent_id}] WARNING: Experimental data file not found at '{resolved_data_path}'.")
            return {
                "experimental_data_summary": f"N/A - Experimental data file not found: {os.path.basename(resolved_data_path)}"}
        try:
            with open(resolved_data_path, 'r', encoding='utf-8') as f:
                data_content = f.read()
            if not data_content.strip():
                log_status(
                    f"[{self.agent_id}] WARNING: Experimental data file '{os.path.basename(resolved_data_path)}' is empty.")
                return {"experimental_data_summary": "N/A - Experimental data file is empty."}

            if current_system_message and not current_system_message.startswith("ERROR:"):
                max_exp_data_len = 10000
                truncated_data_content = data_content
                if len(data_content) > max_exp_data_len:
                    log_status(
                        f"[{self.agent_id}] INFO: Truncating experimental data from {len(data_content)} to {max_exp_data_len} for LLM processing.")
                    truncated_data_content = data_content[:max_exp_data_len]
                prompt = f"Please process and summarize the following experimental data content from file '{os.path.basename(resolved_data_path)}':\n\n---\n{truncated_data_content}\n---"
                summary = call_openai_api(prompt, current_system_message, self.agent_id, self.model_name)
                if summary.startswith("Error:"):
                    log_status(
                        f"[{self.agent_id}] WARNING: Failed to summarize experimental data via LLM: {summary}. Passing raw content.")
                    return {"experimental_data_summary": data_content,
                            "warning": f"LLM summary failed: {summary}. Raw data used."}
                log_status(f"[{self.agent_id}] INFO: Experimental data summarized by LLM.")
                return {"experimental_data_summary": summary}
            else:
                log_status(
                    f"[{self.agent_id}] INFO: Passing raw experimental data content as summary (no LLM processing or SM error).")
                return {"experimental_data_summary": data_content}
        except Exception as e:
            log_status(
                f"[{self.agent_id}] ERROR: Failed to read/process experimental data from {resolved_data_path}: {e}")
            return {"experimental_data_summary": "",
                    "error": f"Failed to read/process experimental data from {os.path.basename(resolved_data_path)}: {e}"}


class KnowledgeIntegratorAgent(Agent):
    def execute(self, inputs: dict) -> dict:
        current_system_message = self.get_formatted_system_message()
        if current_system_message.startswith("ERROR:"):
            return {"integrated_knowledge_brief": "", "error": current_system_message}
        multi_doc_synthesis = inputs.get("multi_doc_synthesis",
                                         "N/A (No multi-document synthesis provided or error upstream).")
        web_research_summary = inputs.get("web_research_summary",
                                          "N/A (No web research summary provided or error upstream).")
        experimental_data_summary = inputs.get("experimental_data_summary",
                                               "N/A (No experimental data provided or error upstream).")
        if inputs.get("multi_doc_synthesis_error") or (
                isinstance(multi_doc_synthesis, str) and multi_doc_synthesis.startswith(
            "Error:")): multi_doc_synthesis = f"[Error in upstream multi-document synthesis: {multi_doc_synthesis}]"
        if inputs.get("web_research_summary_error") or (
                isinstance(web_research_summary, str) and web_research_summary.startswith(
            "Error:")): web_research_summary = f"[Error in upstream web research: {web_research_summary}]"
        if inputs.get("experimental_data_summary_error") or (
                isinstance(experimental_data_summary, str) and experimental_data_summary.startswith(
            "Error:")): experimental_data_summary = f"[Error in upstream experimental data loading: {experimental_data_summary}]"
        max_input_segment_len = 10000
        if len(multi_doc_synthesis) > max_input_segment_len: multi_doc_synthesis = multi_doc_synthesis[
                                                                                   :max_input_segment_len] + "\n[...truncated due to length...]"
        if len(web_research_summary) > max_input_segment_len: web_research_summary = web_research_summary[
                                                                                     :max_input_segment_len] + "\n[...truncated due to length...]"
        prompt = (
            f"Integrate the following information sources into a comprehensive knowledge brief as per your role:\n\n"
            f"1. Cross-Document Synthesis from multiple papers:\n---\n{multi_doc_synthesis}\n---\n\n"
            f"2. Web Research Summary:\n---\n{web_research_summary}\n---\n\n"
            f"3. Experimental Data Summary:\n---\n{experimental_data_summary}\n---\n\n"
            f"Provide the integrated knowledge brief. Focus on synthesizing these distinct pieces of information into novel insights, identifying synergies, conflicts, and critical knowledge gaps.")
        integrated_brief = call_openai_api(prompt, current_system_message, self.agent_id, model_name=self.model_name,
                                           temperature=0.6)
        if integrated_brief.startswith("Error:"): return {"integrated_knowledge_brief": "", "error": integrated_brief}
        return {"integrated_knowledge_brief": integrated_brief}


class HypothesisGeneratorAgent(Agent):
    def execute(self, inputs: dict) -> dict:
        num_hypotheses_to_generate = self.config_params.get("num_hypotheses", 3)
        try:
            num_hypotheses_to_generate = int(num_hypotheses_to_generate)
            if num_hypotheses_to_generate <= 0:
                num_hypotheses_to_generate = 3
                log_status(
                    f"[{self.agent_id}] WARNING: Invalid 'num_hypotheses' config: {self.config_params.get('num_hypotheses')}. Defaulting to 3.")
        except ValueError:
            num_hypotheses_to_generate = 3
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
        max_brief_len = 15000
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
        llm_response_str = call_openai_api(user_prompt, current_system_message, self.agent_id,
                                           model_name=self.model_name)
        if llm_response_str.startswith("Error:"):
            return {"hypotheses_output_blob": llm_response_str, "hypotheses_list": [], "key_opportunities": "",
                    "error": f"LLM call failed: {llm_response_str}"}
        try:
            cleaned_llm_response_str = llm_response_str
            if cleaned_llm_response_str.strip().startswith(
                "```json"): cleaned_llm_response_str = cleaned_llm_response_str.strip()[len("```json"):].strip()
            if cleaned_llm_response_str.strip().endswith(
                "```"): cleaned_llm_response_str = cleaned_llm_response_str.strip()[:-len("```")].strip()
            parsed_output = json.loads(cleaned_llm_response_str)
            key_opportunities = parsed_output.get("key_opportunities", "")
            hypotheses_list = parsed_output.get("hypotheses", [])
            if not isinstance(key_opportunities, str): key_opportunities = str(key_opportunities)
            if not isinstance(hypotheses_list, list): hypotheses_list = []
            hypotheses_list = [str(h).strip() for h in hypotheses_list if
                               isinstance(h, (str, int, float)) and str(h).strip()]
            log_status(
                f"[{self.agent_id}] Successfully parsed hypotheses. Opportunities: '{key_opportunities[:50]}...', Hypotheses count from LLM: {len(hypotheses_list)}")
            return {
                "hypotheses_output_blob": llm_response_str,
                "hypotheses_list": hypotheses_list,
                "key_opportunities": key_opportunities
            }
        except json.JSONDecodeError as e:
            log_status(
                f"[{self.agent_id}] HYPOTHESIS_PARSE_ERROR: Failed to parse JSON from LLM response. Error: {e}. Response (first 500 chars): {llm_response_str[:500]}...")
            return {"hypotheses_output_blob": llm_response_str, "hypotheses_list": [], "key_opportunities": "",
                    "error": f"Failed to parse JSON from LLM: {e}. Response: {llm_response_str[:200]}"}
        except Exception as e:
            log_status(
                f"[{self.agent_id}] HYPOTHESIS_UNEXPECTED_PARSE_ERROR: Error: {e}. Response (first 500 chars): {llm_response_str[:500]}...")
            return {"hypotheses_output_blob": llm_response_str, "hypotheses_list": [], "key_opportunities": "",
                    "error": f"Unexpected error parsing LLM output: {e}"}


class ExperimentDesignerAgent(Agent):
    def execute(self, inputs: dict) -> dict:
        current_system_message = self.get_formatted_system_message()
        if current_system_message.startswith("ERROR:"):
            return {"experiment_designs_list": [], "error": current_system_message}
        hypotheses_list_input = inputs.get("hypotheses_list", [])
        hypotheses_list_had_upstream_error = inputs.get("hypotheses_list_error", False)
        if hypotheses_list_had_upstream_error:
            log_status(f"[{self.agent_id}] INPUT_ERROR: Upstream error in hypotheses_list. Cannot design experiments.")
            return {"experiment_designs_list": [], "error": "Upstream error in hypotheses list."}
        if not isinstance(hypotheses_list_input, list) or not hypotheses_list_input:
            log_status(
                f"[{self.agent_id}] INFO: No hypotheses provided or invalid format for experiment design. Input: {hypotheses_list_input}")
            return {"experiment_designs_list": [], "info": "No valid hypotheses provided to design experiments for."}
        all_designs = []
        for i, hypo_str in enumerate(hypotheses_list_input):
            if not isinstance(hypo_str, str) or not hypo_str.strip():
                log_status(f"[{self.agent_id}] WARNING: Skipping invalid hypothesis at index {i}: '{hypo_str}'")
                all_designs.append({"experiment_design": "", "hypothesis_processed": str(hypo_str),
                                    "error": "Invalid or empty hypothesis string."})
                continue
            log_status(f"[{self.agent_id}] Designing experiment for hypothesis {i + 1}: '{hypo_str[:100]}...'")
            prompt = f"Design a detailed, feasible, and rigorous experimental protocol for the following hypothesis:\n\nHypothesis: \"{hypo_str}\"\n\nAs per your role, include sections like Objective, Methodology & Apparatus, Step-by-step Procedure, Variables & Controls, Data Collection & Analysis, Expected Outcomes & Success Criteria, Potential Challenges & Mitigation, and Ethical Considerations (if applicable)."
            design = call_openai_api(prompt, current_system_message, self.agent_id, model_name=self.model_name)
            design_output_single = {"hypothesis_processed": hypo_str}
            if design.startswith("Error:"):
                design_output_single["experiment_design"] = ""
                design_output_single["error"] = design
            else:
                design_output_single["experiment_design"] = design
            all_designs.append(design_output_single)
        log_status(f"[{self.agent_id}] Finished designing experiments for {len(hypotheses_list_input)} hypotheses.")
        return {"experiment_designs_list": all_designs}


class GraphOrchestrator:
    def __init__(self, graph_definition_from_config):
        self.graph_definition = graph_definition_from_config
        self.agents = {}
        self.adjacency_list = defaultdict(list)
        self.node_order = []
        self._build_graph_and_determine_order()
        self._initialize_agents()

    def _build_graph_and_determine_order(self):
        if 'nodes' not in self.graph_definition or 'edges' not in self.graph_definition:
            log_status("[GraphOrchestrator] ERROR: Graph definition in config must contain 'nodes' and 'edges'.")
            raise ValueError("Graph definition in config must contain 'nodes' and 'edges'.")
        node_ids = {node['id'] for node in self.graph_definition['nodes']}
        if not node_ids:
            log_status("[GraphOrchestrator] WARNING: No nodes defined in the graph.")
            self.node_order = []
            return
        in_degree = {node_id: 0 for node_id in node_ids}
        for edge in self.graph_definition.get('edges', []):
            from_node, to_node = edge.get('from'), edge.get('to')
            if from_node not in node_ids or to_node not in node_ids:
                log_status(f"[GraphOrchestrator] ERROR: Edge references undefined node: {from_node} -> {to_node}")
                raise ValueError(f"Edge references undefined node: {from_node} -> {to_node}")
            self.adjacency_list[from_node].append(to_node)
            in_degree[to_node] += 1
        queue = deque([node_id for node_id in node_ids if in_degree[node_id] == 0])
        self.node_order = []
        while queue:
            u = queue.popleft()
            self.node_order.append(u)
            for v_neighbor in self.adjacency_list.get(u, []):
                in_degree[v_neighbor] -= 1
                if in_degree[v_neighbor] == 0: queue.append(v_neighbor)
        if len(self.node_order) != len(node_ids):
            processed_nodes = set(self.node_order)
            missing_nodes = node_ids - processed_nodes
            log_status(
                f"[GraphOrchestrator] ERROR: Graph has a cycle or is disconnected. Order: {self.node_order}, Degrees: {in_degree}, Missing/Cyclic nodes: {missing_nodes}")
            raise ValueError(
                f"Graph has a cycle or is disconnected. Processed: {len(self.node_order)}/{len(node_ids)} nodes.")
        log_status(f"[GraphOrchestrator] INFO: Node execution order determined: {self.node_order}")

    def _initialize_agents(self):
        agent_class_map = {
            "PDFLoaderAgent": PDFLoaderAgent, "PDFSummarizerAgent": PDFSummarizerAgent,
            "MultiDocSynthesizerAgent": MultiDocSynthesizerAgent, "WebResearcherAgent": WebResearcherAgent,
            "ExperimentalDataLoaderAgent": ExperimentalDataLoaderAgent,
            "KnowledgeIntegratorAgent": KnowledgeIntegratorAgent,
            "HypothesisGeneratorAgent": HypothesisGeneratorAgent, "ExperimentDesignerAgent": ExperimentDesignerAgent,
        }
        for node_def in self.graph_definition.get('nodes', []):
            agent_id = node_def['id']
            agent_type_name = node_def['type']
            agent_config_params = node_def.get('config', {})
            agent_class = agent_class_map.get(agent_type_name)
            if not agent_class:
                log_status(f"[GraphOrchestrator] ERROR: Unknown agent type: {agent_type_name} for node {agent_id}")
                raise ValueError(f"Unknown agent type: {agent_type_name} for node {agent_id}")
            try:
                self.agents[agent_id] = agent_class(agent_id, agent_type_name, agent_config_params)
            except Exception as e:
                log_status(
                    f"[GraphOrchestrator] ERROR: Failed to initialize agent '{agent_id}' of type '{agent_type_name}': {e}")
                raise

    def run(self, all_pdf_paths: list, experimental_data_file_path: str, project_base_output_dir: str):
        outputs_history = {}
        log_status(
            f"[GraphOrchestrator] Starting INTEGRATED workflow for {len(all_pdf_paths)} PDFs. Output dir: {project_base_output_dir}")

        # STAGE 1: PDF Processing
        all_summaries_for_synthesis = []
        pdf_loader_agent = self.agents.get("pdf_loader_node")
        pdf_summarizer_agent = self.agents.get("pdf_summarizer_node")
        if not pdf_loader_agent:
            log_status("[GraphOrchestrator] ERROR: 'pdf_loader_node' not found. Cannot process PDFs.")
            return {"error": "'pdf_loader_node' not defined or initialized."}
        if not pdf_summarizer_agent:
            log_status("[GraphOrchestrator] ERROR: 'pdf_summarizer_node' not found. Cannot summarize PDFs.")
            return {"error": "'pdf_summarizer_node' not defined or initialized."}

        for i, pdf_path in enumerate(all_pdf_paths):
            log_status(
                f"\n[GraphOrchestrator] Processing PDF {i + 1}/{len(all_pdf_paths)}: {os.path.basename(pdf_path)}")
            load_output = pdf_loader_agent.execute({"pdf_path": pdf_path})
            outputs_history[f"pdf_loader_{os.path.basename(pdf_path)}_{i}"] = load_output
            if load_output.get("error"):
                log_status(
                    f"[GraphOrchestrator] ERROR loading PDF {os.path.basename(pdf_path)}: {load_output['error']}")
                all_summaries_for_synthesis.append(
                    {"summary": f"Error loading {os.path.basename(pdf_path)}: {load_output['error']}",
                     "original_pdf_path": pdf_path, "error": True})
                continue
            summary_output = pdf_summarizer_agent.execute(
                {"pdf_text_content": load_output["pdf_text_content"], "original_pdf_path": pdf_path})
            outputs_history[f"pdf_summarizer_{os.path.basename(pdf_path)}_{i}"] = summary_output
            if summary_output.get("error"):
                log_status(
                    f"[GraphOrchestrator] ERROR summarizing PDF {os.path.basename(pdf_path)}: {summary_output['error']}")
                all_summaries_for_synthesis.append(
                    {"summary": f"Error summarizing {os.path.basename(pdf_path)}: {summary_output['error']}",
                     "original_pdf_path": pdf_path, "error": True})
            else:
                all_summaries_for_synthesis.append(
                    {"summary": summary_output["summary"], "original_pdf_path": pdf_path, "error": False})

        # STAGE 2: Multi-Document Synthesis (Explicit call before main graph loop)
        mds_output_data = {}
        if "multi_doc_synthesizer" in self.agents:
            log_status(f"\n[GraphOrchestrator] EXECUTING_NODE (Special Handling): 'multi_doc_synthesizer'")
            mds_agent = self.agents["multi_doc_synthesizer"]
            mds_inputs = {"all_pdf_summaries": all_summaries_for_synthesis}

            if not all_summaries_for_synthesis:
                log_status(
                    f"[{mds_agent.agent_id}] INPUT_ERROR: No PDF summaries collected for synthesis (list is empty).")
                mds_output_data = {"multi_doc_synthesis_output": "",
                                   "error": "No PDF summaries collected for synthesis."}
            else:
                mds_output_data = mds_agent.execute(mds_inputs)

            outputs_history["multi_doc_synthesizer"] = mds_output_data
            log_status(
                f"[multi_doc_synthesizer] RESULT: {{ {', '.join([f'{k}: {str(v)[:70]}...' for k, v in mds_output_data.items()])} }}")
            if mds_output_data.get("error"):
                log_status(
                    f"[GraphOrchestrator] NODE_EXECUTION_ERROR_REPORTED: multi_doc_synthesizer: {mds_output_data['error']}")
        else:
            log_status(
                "[GraphOrchestrator] WARNING: 'multi_doc_synthesizer' agent not found. Multi-document synthesis step will be skipped.")
            outputs_history["multi_doc_synthesizer"] = {"error": "MultiDocSynthesizerAgent not configured or found.",
                                                        "multi_doc_synthesis_output": ""}

        # STAGE 3: Execute remaining graph nodes based on topological order
        nodes_already_explicitly_handled = {"pdf_loader_node", "pdf_summarizer_node", "multi_doc_synthesizer"}

        for node_id in self.node_order:
            if node_id in nodes_already_explicitly_handled:
                continue
            if node_id not in self.agents:
                log_status(
                    f"[GraphOrchestrator] WARNING: Node '{node_id}' from order not found in initialized agents. Skipping.")
                outputs_history[node_id] = {"error": f"Agent for node '{node_id}' not initialized."}
                continue

            log_status(f"\n[GraphOrchestrator] EXECUTING_NODE: '{node_id}' (Type: {self.agents[node_id].agent_type})")
            current_agent = self.agents[node_id]
            agent_inputs = {}
            has_input_error = False
            for edge_def in self.graph_definition.get('edges', []):
                if edge_def['to'] == node_id:
                    from_node_id = edge_def['from']
                    source_outputs = outputs_history.get(from_node_id)
                    if not source_outputs:
                        log_status(
                            f"[GraphOrchestrator] INPUT_ERROR: Output from source node '{from_node_id}' not found for target '{node_id}'.")
                        data_mapping = edge_def.get("data_mapping", {})
                        for _, target_key in data_mapping.items():
                            agent_inputs[target_key] = f"Error: Input from '{from_node_id}' missing."
                            agent_inputs[f"{target_key}_error"] = True
                        has_input_error = True
                        continue
                    data_mapping = edge_def.get("data_mapping")
                    if not data_mapping:
                        log_status(
                            f"[GraphOrchestrator] WARNING: No data_mapping for edge from '{from_node_id}' to '{node_id}'. Merging all outputs.")
                        agent_inputs.update(source_outputs)
                        if source_outputs.get("error"): has_input_error = True
                    else:
                        for src_key, target_key in data_mapping.items():
                            if src_key in source_outputs:
                                agent_inputs[target_key] = source_outputs[src_key]
                                if source_outputs.get("error") or (
                                        isinstance(source_outputs[src_key], str) and source_outputs[src_key].startswith(
                                        "Error:")):
                                    agent_inputs[f"{target_key}_error"] = True
                                    has_input_error = True
                            else:
                                log_status(
                                    f"[GraphOrchestrator] INPUT_ERROR: Source key '{src_key}' not found in output of '{from_node_id}' for target '{node_id}'.")
                                agent_inputs[target_key] = f"Error: Key '{src_key}' missing from '{from_node_id}'."
                                agent_inputs[f"{target_key}_error"] = True
                                has_input_error = True
            if isinstance(current_agent,
                          ExperimentalDataLoaderAgent) and "experimental_data_file_path" not in agent_inputs:
                if experimental_data_file_path: agent_inputs[
                    "experimental_data_file_path"] = experimental_data_file_path
            log_status(
                f"[{node_id}] INFO: Inputs gathered: {{ {', '.join([f'{k}: {str(v)[:60]}...' for k, v in agent_inputs.items()])} }}")
            node_output = {}
            try:
                node_output = current_agent.execute(agent_inputs)
            except Exception as agent_exec_e:
                detailed_traceback = traceback.format_exc()
                log_status(
                    f"[GraphOrchestrator] NODE_EXECUTION_CRITICAL_ERROR: Node '{node_id}' failed during execute(): {agent_exec_e}\n{detailed_traceback}")
                node_output = {"error": f"Agent execution for '{node_id}' failed critically: {agent_exec_e}"}
            outputs_history[node_id] = node_output
            log_status(
                f"[{node_id}] RESULT: {{ {', '.join([f'{k}: {str(v)[:70]}...' for k, v in node_output.items()])} }}")
            if node_output.get("error"): log_status(
                f"[GraphOrchestrator] NODE_EXECUTION_ERROR_REPORTED: Node '{node_id}': {node_output['error']}")
        self._save_consolidated_outputs(outputs_history, project_base_output_dir)
        log_status("\n[GraphOrchestrator] INFO: INTEGRATED workflow execution completed.")
        return outputs_history

    def _save_consolidated_outputs(self, outputs_history: dict, project_base_output_dir: str):
        log_status(f"[GraphOrchestrator] Saving consolidated outputs to {project_base_output_dir}...")
        output_dirs_config = APP_CONFIG.get("system_variables", {})
        project_output_paths = {}
        if not os.path.exists(project_base_output_dir):
            try:
                os.makedirs(project_base_output_dir, exist_ok=True)
            except OSError as e:
                log_status(f"ERROR creating base project output directory {project_base_output_dir}: {e}")
                return
        for key, subfolder_name in output_dirs_config.items():
            if key.startswith("output_project_") and key.endswith("_folder_name"):
                if not isinstance(subfolder_name, str):
                    log_status(
                        f"[GraphOrchestrator] WARNING: Invalid subfolder name for {key}: {subfolder_name}. Skipping.")
                    continue
                full_path = os.path.join(project_base_output_dir, subfolder_name)
                project_output_paths[key.replace("output_project_", "").replace("_folder_name", "")] = full_path
                if not os.path.exists(full_path):
                    try:
                        os.makedirs(full_path, exist_ok=True)
                    except OSError as e:
                        log_status(f"ERROR creating output subfolder {full_path}: {e}")

        def write_output_file(folder_key, filename, content):
            folder_path = project_output_paths.get(folder_key)
            if content is not None and not (
                    isinstance(content, str) and content.startswith("Error:")) and folder_path and os.path.exists(
                    folder_path):
                try:
                    with open(os.path.join(folder_path, filename), "w", encoding="utf-8") as f:
                        f.write(str(content))
                    log_status(f"Saved '{filename}' to '{folder_path}'")
                except Exception as e:
                    log_status(f"ERROR writing file {os.path.join(folder_path, filename)}: {e}")
            elif not folder_path or not os.path.exists(folder_path):
                log_status(
                    f"WARNING: Output folder for key '{folder_key}' ('{folder_path}') does not exist. Cannot save '{filename}'.")
            elif content is None or (isinstance(content, str) and content.startswith("Error:")):
                log_status(
                    f"INFO: No valid content to save for '{filename}' (content was None, empty or an error string).")

        mds_out = outputs_history.get("multi_doc_synthesizer", {}).get("multi_doc_synthesis_output")
        write_output_file("synthesis", "multi_document_synthesis.txt", mds_out)
        web_research_out = outputs_history.get("web_researcher", {}).get("web_summary")
        write_output_file("synthesis", "web_research_summary.txt", web_research_out)
        exp_data_out = outputs_history.get("experimental_data_loader", {}).get("experimental_data_summary", "N/A")
        if exp_data_out != "N/A": write_output_file("synthesis", "experimental_data_summary.txt", exp_data_out)
        ikb_out = outputs_history.get("knowledge_integrator", {}).get("integrated_knowledge_brief")
        write_output_file("synthesis", "integrated_knowledge_brief.txt", ikb_out)
        hypo_gen_node_out = outputs_history.get("hypothesis_generator", {})
        raw_blob = hypo_gen_node_out.get("hypotheses_output_blob")
        write_output_file("hypotheses", "hypotheses_raw_llm_output.json", raw_blob)
        key_ops = hypo_gen_node_out.get("key_opportunities")
        write_output_file("hypotheses", "key_research_opportunities.txt", key_ops)
        hypo_list = hypo_gen_node_out.get("hypotheses_list", [])
        if hypo_list:
            hypo_list_content = ""
            for i, h in enumerate(hypo_list): hypo_list_content += f"{i + 1}. {h}\n\n"
            write_output_file("hypotheses", "hypotheses_list.txt", hypo_list_content.strip())
        exp_designs_list = outputs_history.get("experiment_designer", {}).get("experiment_designs_list", [])
        exp_path = project_output_paths.get("experiments")
        if exp_designs_list and exp_path and os.path.exists(exp_path):
            for i, design_info in enumerate(exp_designs_list):
                hypo = design_info.get("hypothesis_processed", f"Hypothesis_N/A_{i + 1}")
                design = design_info.get("experiment_design", "")
                err = design_info.get("error")
                safe_hypo_part = "".join(c if c.isalnum() else "_" for c in str(hypo)[:50]).strip('_')
                if not safe_hypo_part: safe_hypo_part = f"hypo_text_{i + 1}"
                fname = f"exp_design_{i + 1}_{safe_hypo_part}.txt"
                file_content = f"Hypothesis: {hypo}\n\n"
                if err: file_content += f"Error in design generation: {err}\n\n"
                file_content += f"Experiment Design:\n{design}\n"
                try:
                    with open(os.path.join(exp_path, fname), "w", encoding="utf-8") as f:
                        f.write(file_content)
                    log_status(f"Saved experiment design '{fname}' to '{exp_path}'")
                except Exception as e:
                    log_status(f"ERROR writing experiment design file {os.path.join(exp_path, fname)}: {e}")
        elif not exp_designs_list:
            log_status("INFO: No experiment designs were generated to save.")


def run_project_orchestration(pdf_file_paths: list, experimental_data_path: str, project_base_output_dir: str,
                              status_update_callback: callable, config_file_path: str = "config.json"):
    set_status_callback(status_update_callback)
    resolved_config_path = config_file_path
    if not os.path.isabs(config_file_path): resolved_config_path = os.path.join(SCRIPT_DIR, config_file_path)
    if not load_app_config(resolved_config_path):
        final_error_msg = f"Critical: Configuration load failed from '{resolved_config_path}'. Orchestration cannot proceed."
        log_status(f"[MainWorkflow] ERROR: {final_error_msg}")
        return {"error": final_error_msg}
    if not APP_CONFIG:
        final_error_msg = "Critical: APP_CONFIG is empty after load attempt. Orchestration cannot proceed."
        log_status(f"[MainWorkflow] ERROR: {final_error_msg}")
        return {"error": final_error_msg}

    # Set OpenAI API key for the SDK globally, if SDK is available
    if SDK_AVAILABLE and 'set_default_openai_key' in globals() and callable(set_default_openai_key):
        sdk_api_key = APP_CONFIG.get("system_variables", {}).get("openai_api_key")
        if sdk_api_key and sdk_api_key not in ["YOUR_OPENAI_API_KEY_NOT_IN_CONFIG", "YOUR_ACTUAL_OPENAI_API_KEY",
                                               "KEY"]:
            try:
                set_default_openai_key(sdk_api_key)
                log_status("[MainWorkflow] INFO: OpenAI API key set for openai-agents SDK.")
            except Exception as e:
                log_status(f"[MainWorkflow] WARNING: Failed to set OpenAI API key for SDK: {e}")
        else:
            log_status(
                "[MainWorkflow] WARNING: Valid OpenAI API key not found in APP_CONFIG to set for SDK. SDK calls might fail if OPENAI_API_KEY env var is not set.")
    elif SDK_AVAILABLE:
        log_status(
            "[MainWorkflow] WARNING: 'set_default_openai_key' function not available from SDK import. SDK calls might fail if OPENAI_API_KEY env var is not set.")

    openai_api_key_check = APP_CONFIG.get("system_variables", {}).get("openai_api_key")  # Renamed for clarity
    if not openai_api_key_check or openai_api_key_check in ["YOUR_OPENAI_API_KEY_NOT_IN_CONFIG",
                                                            "YOUR_ACTUAL_OPENAI_API_KEY", "KEY"]:
        api_key_error_msg = f"OpenAI API key missing or is a placeholder in configuration ('{resolved_config_path}'). Please set a valid API key for custom agent calls."
        log_status(
            f"[MainWorkflow] CONFIG_ERROR: {api_key_error_msg}")  # This is for your direct calls, SDK is separate

    if not pdf_file_paths:
        log_status(
            "[MainWorkflow] INFO: No PDF files provided for processing. Workflow will proceed but may lack PDF-derived input.")
    else:
        for p_path in pdf_file_paths:
            if not os.path.exists(p_path):
                path_error_msg = f"Input PDF not found at '{p_path}'. Please check the path."
                log_status(f"[MainWorkflow] FILE_ERROR: {path_error_msg}")
                return {"error": path_error_msg}
    if not os.path.exists(project_base_output_dir):
        try:
            os.makedirs(project_base_output_dir, exist_ok=True)
            log_status(f"[MainWorkflow] Created project output directory: {project_base_output_dir}")
        except OSError as e:
            dir_error_msg = f"Could not create project output directory '{project_base_output_dir}': {e}"
            log_status(f"[MainWorkflow] DIRECTORY_ERROR: {dir_error_msg}")
            return {"error": dir_error_msg}
    try:
        orchestrator = GraphOrchestrator(APP_CONFIG.get("graph_definition"))
        final_outputs = orchestrator.run(all_pdf_paths=pdf_file_paths,
                                         experimental_data_file_path=experimental_data_path,
                                         project_base_output_dir=project_base_output_dir)
        log_status("[MainWorkflow] INTEGRATED project orchestration finished.")
        return final_outputs
    except ValueError as ve:
        log_status(f"[MainWorkflow] ORCHESTRATION_SETUP_ERROR: {ve}")
        return {"error": f"Orchestration setup failed: {ve}"}
    except Exception as e:
        detailed_traceback = traceback.format_exc()
        log_status(f"[MainWorkflow] UNEXPECTED_ORCHESTRATION_ERROR: Orchestration failed: {e}\n{detailed_traceback}")
        return {"error": f"Unexpected error during orchestration: {e}"}


if __name__ == "__main__":
    CLI_CONFIG_FILENAME = "config_cli_test_integrated.json"
    cli_config_full_path = os.path.join(SCRIPT_DIR, CLI_CONFIG_FILENAME)
    if not os.path.exists(cli_config_full_path):
        print(f"[CLI_Test] '{CLI_CONFIG_FILENAME}' not found. Creating a minimal one for integrated test.")
        dummy_cli_config = {
            "system_variables": {"openai_api_key": "YOUR_OPENAI_API_KEY_IN_CLI_CONFIG",
                                 "output_project_synthesis_folder_name": "project_synthesis_cli",
                                 "output_project_hypotheses_folder_name": "project_hypotheses_cli",
                                 "output_project_experiments_folder_name": "project_experiments_cli",
                                 "default_llm_model": "gpt-3.5-turbo",
                                 "models": {"pdf_summarizer": "gpt-3.5-turbo",
                                            "multi_doc_synthesizer_model": "gpt-3.5-turbo",
                                            "web_research_model": "gpt-4o",
                                            "knowledge_integrator_model": "gpt-3.5-turbo",
                                            "hypothesis_generator": "gpt-3.5-turbo",
                                            "experiment_designer": "gpt-3.5-turbo",
                                            "sdk_planner_model": "gpt-3.5-turbo", "sdk_search_model": "gpt-3.5-turbo",
                                            "sdk_writer_model": "gpt-3.5-turbo"},
                                 "openai_api_timeout_seconds": 180},
            "agent_prompts": {"pdf_summarizer_sm": "Summarize this academic text in 1-2 paragraphs: {text_content}",
                              "multi_doc_synthesizer_sm": "Synthesize these summaries: {all_summaries_text}. Output: Cross-document understanding.",
                              "web_researcher_sm": "This agent uses an SDK for web research.",
                              "experimental_data_loader_sm": "This is experimental data: {data_content}. Present as a structured summary.",
                              "knowledge_integrator_sm": "Integrate: Multi-doc: {multi_doc_synthesis}, Web: {web_research_summary}, ExpData: {experimental_data_summary}. Output: Integrated brief.",
                              "hypothesis_generator_sm": "You are a highly insightful research strategist and innovator. Based on the provided 'integrated knowledge brief' (which consolidates information from multiple papers, web research, and experimental data), your task is to:\n1.  **Identify Key Opportunities:** Briefly highlight the most promising areas for novel research based on the integrated knowledge.\n2.  **Generate Hypotheses:** Propose {num_hypotheses} distinct, novel, and groundbreaking hypotheses that go beyond the current state-of-the-art. These hypotheses should be specific, testable, and aim to open new avenues of research by directly addressing the identified gaps or leveraging the novel connections found in the integrated brief.\n\n**Output Format STRICTLY REQUIRED:** Provide your response as a single JSON object with two keys:\n- `\"key_opportunities\"`: A string containing your brief summary of key research opportunities.\n- `\"hypotheses\"`: A JSON array of strings, where each string is a clearly articulated hypothesis.\n\nExample JSON output structure:\n```json\n{{\n  \"key_opportunities\": \"The integration of X from papers, Y from web, and Z from experiments points to a critical need to investigate C.\",\n  \"hypotheses\": [\n    \"Hypothesis 1: ...\",\n    \"Hypothesis 2: ...\"\n  ]\n}}\n```\nEnsure the entire output is a valid JSON object.",
                              "experiment_designer_sm": "Design experiment for: {hypothesis}",
                              "sdk_planner_sm": "Plan 2-3 web searches for query: {query}",
                              "sdk_searcher_sm": "Search web for: {search_term}. Reason: {reason}. Summarize results concisely.",
                              "sdk_writer_sm": "Write a brief report from query: {query} and search results: {search_results}."},
            "graph_definition": {"nodes": [
                {"id": "pdf_loader_node", "type": "PDFLoaderAgent", "config": {"description": "Loads text from PDF."}},
                {"id": "pdf_summarizer_node", "type": "PDFSummarizerAgent",
                 "config": {"model_key": "pdf_summarizer", "system_message_key": "pdf_summarizer_sm"}},
                {"id": "multi_doc_synthesizer", "type": "MultiDocSynthesizerAgent",
                 "config": {"model_key": "multi_doc_synthesizer_model",
                            "system_message_key": "multi_doc_synthesizer_sm"}},
                {"id": "web_researcher", "type": "WebResearcherAgent",
                 "config": {"model_key": "web_research_model", "system_message_key": "web_researcher_sm"}},
                {"id": "experimental_data_loader", "type": "ExperimentalDataLoaderAgent",
                 "config": {"system_message_key": "experimental_data_loader_sm"}},
                {"id": "knowledge_integrator", "type": "KnowledgeIntegratorAgent",
                 "config": {"model_key": "knowledge_integrator_model",
                            "system_message_key": "knowledge_integrator_sm"}},
                {"id": "hypothesis_generator", "type": "HypothesisGeneratorAgent",
                 "config": {"model_key": "hypothesis_generator", "system_message_key": "hypothesis_generator_sm",
                            "num_hypotheses": 3}},
                {"id": "experiment_designer", "type": "ExperimentDesignerAgent",
                 "config": {"model_key": "experiment_designer", "system_message_key": "experiment_designer_sm"}}],
                "edges": [
                    {"from": "multi_doc_synthesizer", "to": "web_researcher",
                     "data_mapping": {"multi_doc_synthesis_output": "cross_document_understanding"}},
                    {"from": "multi_doc_synthesizer", "to": "knowledge_integrator",
                     "data_mapping": {"multi_doc_synthesis_output": "multi_doc_synthesis"}},
                    {"from": "web_researcher", "to": "knowledge_integrator",
                     "data_mapping": {"web_summary": "web_research_summary"}},
                    {"from": "experimental_data_loader", "to": "knowledge_integrator",
                     "data_mapping": {"experimental_data_summary": "experimental_data_summary"}},
                    {"from": "knowledge_integrator", "to": "hypothesis_generator",
                     "data_mapping": {"integrated_knowledge_brief": "integrated_knowledge_brief"}},
                    {"from": "hypothesis_generator", "to": "experiment_designer",
                     "data_mapping": {"hypotheses_list": "hypotheses_list"}}
                ]}}
        try:
            with open(cli_config_full_path, 'w') as f:
                json.dump(dummy_cli_config, f, indent=2)
            print(
                f"[CLI_Test] Created '{cli_config_full_path}'. IMPORTANT: Please review and add a valid OpenAI API key.")
        except Exception as e:
            print(f"[CLI_Test] ERROR writing dummy CLI config: {e}")
            CLI_CONFIG_FILENAME = "config.json"
            cli_config_full_path = os.path.join(SCRIPT_DIR, CLI_CONFIG_FILENAME)
            print(f"[CLI_Test] Falling back to main '{CLI_CONFIG_FILENAME}'.")
    set_status_callback(print)
    if not load_app_config(config_path=cli_config_full_path):
        print(f"[CLI_Test] CRITICAL: Failed to load CLI config '{cli_config_full_path}'. Exiting.")
        exit(1)
    cli_api_key = APP_CONFIG.get("system_variables", {}).get("openai_api_key")
    if not cli_api_key or cli_api_key == "YOUR_OPENAI_API_KEY_IN_CLI_CONFIG": print(
        f"[CLI_Test] WARNING: OpenAI API key in '{cli_config_full_path}' is a placeholder. LLM calls will likely fail.")
    cli_test_input_dir = os.path.join(SCRIPT_DIR, "cli_test_multi_input_pdfs")
    cli_test_exp_data_dir = os.path.join(SCRIPT_DIR, "cli_test_experimental_data")
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    cli_test_project_output_dir = os.path.join(SCRIPT_DIR, f"cli_test_integrated_project_output_{timestamp}")
    for p_dir in [cli_test_input_dir, cli_test_exp_data_dir, cli_test_project_output_dir]:
        if not os.path.exists(p_dir):
            try:
                os.makedirs(p_dir, exist_ok=True)
            except OSError as e:
                print(f"[CLI_Test] ERROR creating directory {p_dir}: {e}. Exiting."); exit(1)
    pdf_paths_for_test = []
    if REPORTLAB_AVAILABLE:
        for i in range(2):
            pdf_name = f"dummy_paper_cli_{i + 1}.pdf"
            full_pdf_path = os.path.join(cli_test_input_dir, pdf_name)
            try:
                c = canvas.Canvas(full_pdf_path, pagesize=letter)
                text_obj = c.beginText(inch, 10 * inch);
                text_obj.setFont("Helvetica", 10)
                text_obj.textLine(f"CLI Test: Dummy PDF Document {i + 1}.");
                text_obj.textLine(f"This is page 1 of dummy document {i + 1}.")
                text_obj.textLine("It contains some sample text for testing PDF processing and summarization.")
                if i == 0:
                    text_obj.textLine(
                        "This document discusses advancements in AI-driven research methodologies."); text_obj.textLine(
                        "Keywords: artificial intelligence, machine learning, research automation, LLMs.")
                else:
                    text_obj.textLine(
                        "This document focuses on the challenges of multi-modal data integration."); text_obj.textLine(
                        "Keywords: data fusion, text analysis, image processing, knowledge graphs.")
                c.drawText(text_obj);
                c.showPage()
                text_obj = c.beginText(inch, 10 * inch);
                text_obj.setFont("Helvetica", 10)
                text_obj.textLine(f"Page 2 of dummy document {i + 1}.");
                text_obj.textLine("Further details and elaborations are usually found here.")
                c.drawText(text_obj);
                c.save()
                pdf_paths_for_test.append(full_pdf_path)
                print(f"[CLI_Test] Created dummy PDF: {full_pdf_path}")
            except Exception as e:
                print(f"[CLI_Test] ERROR creating dummy PDF {pdf_name}: {e}")
    else:
        print(
            "[CLI_Test] WARNING: reportlab library not found. Cannot create dummy PDFs for testing. Please create them manually in 'cli_test_multi_input_pdfs' or install reportlab.")
        for i in range(2):
            pdf_name = f"dummy_paper_cli_{i + 1}.pdf";
            full_pdf_path = os.path.join(cli_test_input_dir, pdf_name)
            if os.path.exists(full_pdf_path): pdf_paths_for_test.append(full_pdf_path); print(
                f"[CLI_Test] Using existing dummy PDF: {full_pdf_path}")
        if not pdf_paths_for_test: print(
            "[CLI_Test] No dummy PDFs found or created. PDF processing steps will likely fail.")
    exp_data_filename = "dummy_experimental_results_cli.txt"
    exp_data_full_path = os.path.join(cli_test_exp_data_dir, exp_data_filename)
    try:
        with open(exp_data_full_path, "w", encoding="utf-8") as f:
            f.write("Experimental Results Summary (CLI Test):\n");
            f.write("Experiment Alpha: Achieved 95% accuracy in classification task.\n")
            f.write("Experiment Beta: Showed a 20ms reduction in average latency.\n");
            f.write("Conclusion: The new model variant shows significant improvements.")
        print(f"[CLI_Test] Created dummy experimental data: {exp_data_full_path}")
    except Exception as e:
        print(f"[CLI_Test] ERROR creating dummy experimental data: {e}"); exp_data_full_path = None
    print(f"\n[CLI_Test] --- Running INTEGRATED project orchestration via CLI ---")
    print(
        f"[CLI_Test] Input PDFs ({len(pdf_paths_for_test)}): {cli_test_input_dir if pdf_paths_for_test else 'None (will use empty list)'}")
    print(f"[CLI_Test] Experimental data: {exp_data_full_path if exp_data_full_path else 'N/A'}")
    print(f"[CLI_Test] Project outputs will be in: {cli_test_project_output_dir}")
    print(f"[CLI_Test] Using config: {cli_config_full_path}\n")
    if not SDK_AVAILABLE: print(
        "[CLI_Test] WARNING: OpenAI Agents SDK is not installed. WebResearcherAgent will not function as intended.\nPlease run 'pip install openai-agents pydantic typing-extensions' to test full capabilities.")
    results = run_project_orchestration(pdf_file_paths=pdf_paths_for_test, experimental_data_path=exp_data_full_path,
                                        project_base_output_dir=cli_test_project_output_dir,
                                        status_update_callback=print, config_file_path=cli_config_full_path)
    print("\n" + "=" * 30 + " CLI INTEGRATED TEST FINAL RESULTS " + "=" * 30)
    if results and results.get("error"):
        print(f"CLI Test Run completed with an error: {results['error']}")
    elif results:
        print(f"CLI Test Run completed. Key outputs are in '{cli_test_project_output_dir}'.")
        final_ikb = results.get("knowledge_integrator", {}).get("integrated_knowledge_brief")
        if final_ikb:
            print(f"\nSnippet of Integrated Knowledge Brief:\n{str(final_ikb)[:300]}...")
        else:
            print("Integrated Knowledge Brief was not generated or found in results.")
    else:
        print("CLI Test Run did not produce a results dictionary (may have failed very early).")
    print("=" * (60 + len(" CLI INTEGRATED TEST FINAL RESULTS ")))
    print("\n--- Multi-Agent LLM System Backend CLI Integrated Test Finished ---")

