import os
import time
import shutil
import json
from collections import defaultdict, deque

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


# --- Utility Functions ---
def load_app_config(config_path="config.json"):
    global APP_CONFIG, OpenAI, PyPDF2, REPORTLAB_AVAILABLE, canvas, letter, inch, openai_errors
    OpenAI = None  # Reset on each load
    PyPDF2 = None
    REPORTLAB_AVAILABLE = False
    canvas, letter, inch = None, None, None
    openai_errors = None

    resolved_config_path = config_path
    if not os.path.isabs(config_path):
        resolved_config_path = os.path.join(SCRIPT_DIR, config_path)

    log_status(f"[AppConfig] Attempting to load configuration from resolved path: '{resolved_config_path}'")
    try:
        with open(resolved_config_path, 'r', encoding='utf-8') as f:  # Added encoding
            APP_CONFIG = json.load(f)
        log_status(f"[AppConfig] Successfully loaded configuration from '{resolved_config_path}'.")

        # Dynamically import libraries
        try:
            from openai import OpenAI as OpenAI_lib, APIConnectionError, APITimeoutError, RateLimitError, \
                AuthenticationError, BadRequestError
            OpenAI = OpenAI_lib
            openai_errors = {
                "APIConnectionError": APIConnectionError, "APITimeoutError": APITimeoutError,
                "RateLimitError": RateLimitError, "AuthenticationError": AuthenticationError,
                "BadRequestError": BadRequestError
            }
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
    APP_CONFIG = {}  # Ensure APP_CONFIG is empty on any failure
    return False


def get_model_name(model_key=None):
    if not APP_CONFIG: return "gpt-4o"  # Fallback
    models_config = APP_CONFIG.get("system_variables", {}).get("models", {})
    if model_key and model_key in models_config:
        return models_config[model_key]
    return APP_CONFIG.get("system_variables", {}).get("default_llm_model", "gpt-4o")


def get_prompt_text(prompt_key):
    # MODIFICATION: Handle prompt_key being None gracefully
    if prompt_key is None:
        # log_status(f"[AppConfig] INFO: Prompt key is None. No specific system message will be loaded by key.")
        return ""  # Return empty string, call_openai_api will use its default system message.

    if not APP_CONFIG: return f"ERROR: Config not loaded, prompt key '{prompt_key}' unavailable."
    prompts_config = APP_CONFIG.get("agent_prompts", {})

    if prompt_key not in prompts_config:
        log_status(f"[AppConfig] ERROR: Prompt key '{prompt_key}' not found in agent_prompts.")
        return f"ERROR: Prompt key '{prompt_key}' not found."  # This error will be caught by Agent.__init__

    prompt_text = prompts_config.get(prompt_key)
    if prompt_text is None:  # Key exists but value is null
        log_status(f"[AppConfig] WARNING: Prompt key '{prompt_key}' has null value in config. Returning empty string.")
        return ""
    return prompt_text


def call_openai_api(prompt, system_message="You are a helpful assistant.", agent_name="LLM", model_name=None,
                    temperature=0.5):
    chosen_model = model_name if model_name else get_model_name()

    # If system_message became an error string from get_prompt_text, handle it.
    if isinstance(system_message, str) and system_message.startswith("ERROR:"):
        log_status(f"[{agent_name}] LLM_CALL_ERROR: Invalid system message provided: {system_message}")
        return f"Error: Invalid system message for agent {agent_name} due to: {system_message}"

    # If system_message is empty (e.g. from a None prompt_key), use the default.
    effective_system_message = system_message if system_message else "You are a helpful assistant."

    prompt_display_snippet = prompt[:150].replace('\n', ' ')
    log_status(
        f"[{agent_name}] LLM_CALL_START: Model='{chosen_model}', Temp='{temperature}', SystemMessage='{effective_system_message[:70]}...', Prompt(start): '{prompt_display_snippet}...'")

    if not OpenAI:
        return f"Error: OpenAI library not available for model {chosen_model}."
    if not APP_CONFIG:
        return f"Error: Application configuration not loaded for model {chosen_model}."

    api_key_to_use = APP_CONFIG.get("system_variables", {}).get("openai_api_key")
    api_timeout_seconds = APP_CONFIG.get("system_variables", {}).get("openai_api_timeout_seconds", 60)

    if not api_key_to_use or api_key_to_use in ["YOUR_OPENAI_API_KEY_NOT_IN_CONFIG", "YOUR_ACTUAL_OPENAI_API_KEY",
                                                "KEY"]:
        return f"Error: OpenAI API key not configured for model {chosen_model}."

    try:
        client = OpenAI(api_key=api_key_to_use, timeout=api_timeout_seconds)
        response = client.chat.completions.create(
            model=chosen_model,
            messages=[{"role": "system", "content": effective_system_message}, {"role": "user", "content": prompt}],
            temperature=temperature
        )

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
                    if hasattr(e, 'response') and hasattr(e.response, 'json'):
                        try:
                            err_json = e.response.json()
                            if 'error' in err_json and 'message' in err_json['error']:
                                error_detail = err_json['error']['message']
                        except:
                            pass
                    return f"Error: OpenAI API {err_name} for {chosen_model}: {error_detail}"

        log_status(f"[{agent_name}] LLM_ERROR (General {error_type_name}): API call with {chosen_model} failed: {e}")
        return f"Error: API call with {chosen_model} failed ({error_type_name}): {e}"


# --- Base Agent Definition ---
class Agent:
    def __init__(self, agent_id, agent_type, config_params=None):
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.config_params = config_params if config_params else {}
        self.model_name = get_model_name(self.config_params.get("model_key"))

        system_message_key = self.config_params.get("system_message_key")  # This can be None
        self.system_message = get_prompt_text(system_message_key)  # Handles None key by returning ""
        # Returns "ERROR:..." if key exists but not found

        description = self.config_params.get("description", "No description provided.")
        log_status(
            f"[AgentInit] Created Agent: ID='{self.agent_id}', Type='{self.agent_type}', PrimaryModel='{self.model_name}'. SystemMsgKey='{system_message_key}'. Desc: {description}")

        # Log issues with system message resolution
        if self.system_message.startswith("ERROR:"):  # This means get_prompt_text returned an error string
            log_status(
                f"[AgentInit] CRITICAL_WARNING: System message for Agent ID='{self.agent_id}' (Key: '{system_message_key}') could not be resolved: {self.system_message}")
        elif system_message_key and not self.system_message:  # Key was provided, but prompt text is empty (e.g. key exists in JSON but value is "" or null)
            log_status(
                f"[AgentInit] WARNING: System message for Agent ID='{self.agent_id}' (Key: '{system_message_key}') is empty.")
        # If system_message_key is None, self.system_message will be "" (empty string), which is fine.

    def execute(self, inputs: dict) -> dict:
        # MODIFICATION: Base execute only performs pre-checks.
        # If system_message is an "ERROR:..." string, it means get_prompt_text failed to find a *specified* key.
        # An empty self.system_message (from a None key or empty prompt) is NOT an error for the base class.
        if self.system_message.startswith("ERROR:"):
            return {
                "error": f"Agent {self.agent_id} cannot execute due to configuration error for system message (key: '{self.config_params.get('system_message_key')}'): {self.system_message}"}
        return None  # Signifies base checks passed, subclass should proceed.


# --- Specific Agent Implementations ---
class PDFLoaderAgent(Agent):
    def execute(self, inputs: dict) -> dict:
        # MODIFICATION: Call super().execute() for pre-checks
        base_pre_check_result = super().execute(inputs)
        if base_pre_check_result:  # If base class found an issue (e.g. bad system message for an LLM-based agent)
            return base_pre_check_result  # Propagate that error
            # Note: PDFLoaderAgent doesn't use an LLM, so a system message error is less critical for its core task,
            # but we keep the pattern for consistency. It won't try to call_openai_api with a bad system_message.

        pdf_path = inputs.get("pdf_path")
        if not pdf_path:
            return {"pdf_text_content": "", "error": "PDF path not provided."}
        log_status(f"[{self.agent_id}] PDF_LOAD_START: Path='{pdf_path}'")
        if not PyPDF2:
            return {"pdf_text_content": "", "error": "PyPDF2 library not available."}
        if not os.path.exists(pdf_path):
            return {"pdf_text_content": "", "error": f"PDF file not found: {pdf_path}"}
        try:
            if os.path.getsize(pdf_path) == 0:
                return {"pdf_text_content": "", "error": f"PDF file is empty: {pdf_path}"}
        except OSError as oe:
            return {"pdf_text_content": "", "error": f"Could not access file for size check: {pdf_path}, {oe}"}
        text_content = ""
        try:
            with open(pdf_path, 'rb') as pdf_file_obj:
                pdf_reader = PyPDF2.PdfReader(pdf_file_obj)
                if pdf_reader.is_encrypted:
                    if pdf_reader.decrypt('') == 0:  # 0 means decryption failed
                        log_status(
                            f"[{self.agent_id}] PDF_LOAD_ERROR: Failed to decrypt PDF '{os.path.basename(pdf_path)}'.")
                        return {"pdf_text_content": "",
                                "error": f"Failed to decrypt PDF: {os.path.basename(pdf_path)}."}
                    log_status(f"[{self.agent_id}] PDF_LOAD_INFO: PDF '{os.path.basename(pdf_path)}' decrypted.")
                for page_obj in pdf_reader.pages:
                    text_content += page_obj.extract_text() or ""  # Ensure None is handled
            if not text_content.strip():
                log_status(
                    f"[{self.agent_id}] PDF_LOAD_WARNING: No text extracted from '{os.path.basename(pdf_path)}'.")
            return {"pdf_text_content": text_content, "original_pdf_path": pdf_path}
        except Exception as e:
            log_status(f"[{self.agent_id}] PDF_LOAD_ERROR: PDF extraction failed for {pdf_path}: {e}")
            return {"pdf_text_content": "", "error": f"PDF extraction failed for {pdf_path}: {e}"}


class PDFSummarizerAgent(Agent):
    def execute(self, inputs: dict) -> dict:
        base_pre_check_result = super().execute(inputs)
        if base_pre_check_result:
            return base_pre_check_result

        pdf_text_content = inputs.get("pdf_text_content")
        original_pdf_path = inputs.get("original_pdf_path", "Unknown PDF")

        if inputs.get("pdf_text_content_error") or not pdf_text_content or \
                (isinstance(pdf_text_content, str) and pdf_text_content.startswith("Error:")):
            error_msg = f"Invalid text content for summarization from {original_pdf_path}. Upstream error: {inputs.get('error', pdf_text_content)}"
            return {"summary": "", "error": error_msg, "original_pdf_path": original_pdf_path}

        max_len = self.config_params.get("max_input_length", 15000)
        if len(pdf_text_content) > max_len:
            pdf_text_content = pdf_text_content[:max_len]
        prompt = f"Please summarize the following academic text from document '{os.path.basename(original_pdf_path)}':\n\n---\n{pdf_text_content}\n---"
        summary = call_openai_api(prompt, self.system_message, self.agent_id, model_name=self.model_name)
        if summary.startswith("Error:"):
            return {"summary": "", "error": summary, "original_pdf_path": original_pdf_path}
        return {"summary": summary, "original_pdf_path": original_pdf_path}


class MultiDocSynthesizerAgent(Agent):
    def execute(self, inputs: dict) -> dict:
        base_pre_check_result = super().execute(inputs)
        if base_pre_check_result:
            return base_pre_check_result

        summaries_list = inputs.get("all_pdf_summaries")
        if not summaries_list or not isinstance(summaries_list, list):
            return {"multi_doc_synthesis_output": "", "error": "No PDF summaries provided or input is not a list."}

        valid_summaries = [s for s in summaries_list if
                           s.get("summary") and not s.get("summary", "").startswith("Error:")]
        if not valid_summaries:
            return {"multi_doc_synthesis_output": "", "error": "No valid PDF summaries available for synthesis."}

        formatted_summaries = []
        for i, item in enumerate(valid_summaries):
            pdf_name = os.path.basename(item.get("original_pdf_path", f"Document {i + 1}"))
            formatted_summaries.append(f"Summary from '{pdf_name}':\n{item['summary']}\n---")

        combined_summaries_text = "\n\n".join(formatted_summaries)
        prompt = f"Synthesize the following collection of summaries from multiple academic documents:\n\n{combined_summaries_text}\n\nProvide a coherent 'cross-document understanding' as per your role description."

        synthesis_output = call_openai_api(prompt, self.system_message, self.agent_id, model_name=self.model_name,
                                           temperature=0.6)
        if synthesis_output.startswith("Error:"):
            return {"multi_doc_synthesis_output": "", "error": synthesis_output}
        return {"multi_doc_synthesis_output": synthesis_output}


class WebResearcherAgent(Agent):
    def execute(self, inputs: dict) -> dict:
        base_pre_check_result = super().execute(inputs)
        if base_pre_check_result:
            return base_pre_check_result

        input_text_for_research = inputs.get("cross_document_understanding") or inputs.get("pdf_summary")

        if not input_text_for_research or (
                isinstance(input_text_for_research, str) and input_text_for_research.startswith("Error:")):
            error_msg = f"Invalid input text for web research. Upstream error: {inputs.get('error', input_text_for_research)}"
            return {"web_summary": "", "error": error_msg}

        max_input_len = self.config_params.get("max_snippet_input_length", 4000)
        truncated_input_text = input_text_for_research
        if len(input_text_for_research) > max_input_len:
            truncated_input_text = input_text_for_research[:max_input_len] + "..."

        user_prompt = (
            f"Based on the following synthesized understanding from one or more documents:\n---\n{truncated_input_text}\n---\n\n"
            f"Please find complementary information, recent developments, and broader context, as if performing a comprehensive web search. "
            f"Synthesize your findings into a coherent overview. If you cannot perform a live search, clearly state so and use your training data."
        )
        web_summary_output = call_openai_api(user_prompt, self.system_message, self.agent_id,
                                             model_name=self.model_name)
        if web_summary_output.startswith("Error:"):
            return {"web_summary": "", "error": web_summary_output}
        return {"web_summary": web_summary_output}


class ExperimentalDataLoaderAgent(Agent):
    def execute(self, inputs: dict) -> dict:
        base_pre_check_result = super().execute(inputs)
        if base_pre_check_result:
            return base_pre_check_result

        data_file_path = inputs.get("experimental_data_file_path")
        if not data_file_path:
            log_status(f"[{self.agent_id}] INFO: No experimental data file path provided. Proceeding without it.")
            return {"experimental_data_summary": "N/A - No experimental data file provided."}

        resolved_data_path = data_file_path
        # Path resolution: GUI should provide absolute. CLI test should construct absolute or relative to SCRIPT_DIR.
        if not os.path.isabs(data_file_path):
            # This logic might need adjustment based on how relative paths for exp_data are intended
            # For now, assume it's relative to CWD if not absolute.
            pass

        if not os.path.exists(resolved_data_path):
            log_status(f"[{self.agent_id}] WARNING: Experimental data file not found at '{resolved_data_path}'.")
            return {"experimental_data_summary": f"N/A - Experimental data file not found: {resolved_data_path}"}

        try:
            with open(resolved_data_path, 'r', encoding='utf-8') as f:
                data_content = f.read()
            if not data_content.strip():
                return {"experimental_data_summary": "N/A - Experimental data file is empty."}

            if self.system_message and not self.system_message.startswith("ERROR:"):
                prompt = f"Please process and summarize the following experimental data content:\n\n---\n{data_content[:10000]}\n---"
                summary = call_openai_api(prompt, self.system_message, self.agent_id, self.model_name)
                if summary.startswith("Error:"):
                    return {"experimental_data_summary": data_content,
                            "error": f"Failed to summarize experimental data via LLM: {summary}"}
                return {"experimental_data_summary": summary}
            else:
                return {"experimental_data_summary": data_content}

        except Exception as e:
            return {"experimental_data_summary": "",
                    "error": f"Failed to read/process experimental data from {resolved_data_path}: {e}"}


class KnowledgeIntegratorAgent(Agent):
    def execute(self, inputs: dict) -> dict:
        base_pre_check_result = super().execute(inputs)
        if base_pre_check_result:
            return base_pre_check_result

        multi_doc_synthesis = inputs.get("multi_doc_synthesis", "N/A or error in upstream.")
        web_research_summary = inputs.get("web_research_summary", "N/A or error in upstream.")
        experimental_data_summary = inputs.get("experimental_data_summary", "N/A")

        if isinstance(multi_doc_synthesis, str) and multi_doc_synthesis.startswith("Error:"):
            multi_doc_synthesis = f"[Upstream multi-doc synthesis error: {multi_doc_synthesis}]"
        if isinstance(web_research_summary, str) and web_research_summary.startswith("Error:"):
            web_research_summary = f"[Upstream web research error: {web_research_summary}]"
        if isinstance(experimental_data_summary, str) and experimental_data_summary.startswith("Error:"):
            experimental_data_summary = f"[Upstream experimental data error: {experimental_data_summary}]"

        prompt = (
            f"Integrate the following information sources into a comprehensive knowledge brief as per your role:\n\n"
            f"1. Cross-Document Synthesis from multiple papers:\n---\n{multi_doc_synthesis}\n---\n\n"
            f"2. Web Research Summary:\n---\n{web_research_summary}\n---\n\n"
            f"3. Experimental Data Summary:\n---\n{experimental_data_summary}\n---\n\n"
            f"Provide the integrated knowledge brief."
        )
        integrated_brief = call_openai_api(prompt, self.system_message, self.agent_id, model_name=self.model_name,
                                           temperature=0.6)
        if integrated_brief.startswith("Error:"):
            return {"integrated_knowledge_brief": "", "error": integrated_brief}
        return {"integrated_knowledge_brief": integrated_brief}


class HypothesisGeneratorAgent(Agent):
    def execute(self, inputs: dict) -> dict:
        base_pre_check_result = super().execute(inputs)
        if base_pre_check_result:
            return base_pre_check_result

        integrated_knowledge_brief = inputs.get("integrated_knowledge_brief")
        if not integrated_knowledge_brief or (
                isinstance(integrated_knowledge_brief, str) and integrated_knowledge_brief.startswith("Error:")):
            error_msg = f"Invalid or missing integrated knowledge brief. Upstream error: {inputs.get('error', integrated_knowledge_brief)}"
            return {"hypotheses_output_blob": "", "hypotheses_list": [],
                    "key_opportunities": "", "error": error_msg}

        user_prompt = (
            f"Based on the following 'Integrated Knowledge Brief':\n\n---\n{integrated_knowledge_brief}\n---\n\n"
            "Please provide your analysis, key research opportunities, and proposed hypotheses strictly in the specified JSON format."
        )
        llm_response_str = call_openai_api(user_prompt, self.system_message, self.agent_id, model_name=self.model_name)

        if llm_response_str.startswith("Error:"):
            return {"hypotheses_output_blob": llm_response_str, "hypotheses_list": [],
                    "key_opportunities": "", "error": f"LLM call failed: {llm_response_str}"}
        try:
            cleaned_llm_response_str = llm_response_str
            if cleaned_llm_response_str.startswith("```json"):
                cleaned_llm_response_str = cleaned_llm_response_str[len("```json"):].strip()
            if cleaned_llm_response_str.endswith("```"):
                cleaned_llm_response_str = cleaned_llm_response_str[:-len("```")].strip()

            parsed_output = json.loads(cleaned_llm_response_str)
            key_opportunities = parsed_output.get("key_opportunities", "")
            hypotheses_list = parsed_output.get("hypotheses", [])

            if not isinstance(key_opportunities, str): key_opportunities = ""
            if not isinstance(hypotheses_list, list): hypotheses_list = []
            hypotheses_list = [str(h) for h in hypotheses_list if isinstance(h, str) and h.strip()]

            return {
                "hypotheses_output_blob": llm_response_str,
                "hypotheses_list": hypotheses_list,
                "key_opportunities": key_opportunities
            }
        except Exception as e:  # Catch JSONDecodeError and other potential errors
            log_status(
                f"[{self.agent_id}] HYPOTHESIS_ERROR: Failed to parse JSON from LLM response. Error: {e}. Response was: {llm_response_str[:500]}...")
            return {"hypotheses_output_blob": llm_response_str, "hypotheses_list": [],
                    "key_opportunities": "", "error": f"Failed to parse JSON from LLM: {e}"}


class ExperimentDesignerAgent(Agent):
    def execute(self, inputs: dict) -> dict:
        base_pre_check_result = super().execute(inputs)
        if base_pre_check_result:
            return base_pre_check_result

        hypothesis = inputs.get("hypothesis")
        if inputs.get("hypothesis_error") or not hypothesis or \
                (isinstance(hypothesis, str) and (hypothesis.startswith("Error:") or not hypothesis.strip())):
            error_msg = f"Invalid hypothesis for experiment design: {inputs.get('error', hypothesis)}"
            return {"experiment_design": "", "hypothesis_processed": str(hypothesis), "error": error_msg}

        prompt = f"Design a detailed experiment for this hypothesis:\n\nHypothesis: \"{hypothesis}\"\n\nInclude standard sections."
        design = call_openai_api(prompt, self.system_message, self.agent_id, model_name=self.model_name)
        if design.startswith("Error:"):
            return {"experiment_design": "", "hypothesis_processed": hypothesis, "error": design}
        return {"experiment_design": design, "hypothesis_processed": hypothesis}


# --- Graph Orchestrator ---
# ... (GraphOrchestrator and _save_consolidated_outputs remain the same as in multi_agent_llm_system_v4_integrated)
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
            raise ValueError("Graph definition in config must contain 'nodes' and 'edges'.")
        node_ids = {node['id'] for node in self.graph_definition['nodes']}
        if not node_ids:
            self.node_order = []
            return
        in_degree = {node_id: 0 for node_id in node_ids}
        for edge in self.graph_definition.get('edges', []):
            if edge.get('from') not in node_ids or edge.get('to') not in node_ids:
                raise ValueError(f"Edge references undefined node: {edge.get('from')} -> {edge.get('to')}")
            self.adjacency_list[edge['from']].append(edge['to'])
            in_degree[edge['to']] += 1
        queue = deque([node_id for node_id in node_ids if in_degree[node_id] == 0])
        self.node_order = []
        while queue:
            u = queue.popleft()
            self.node_order.append(u)
            for v_neighbor in self.adjacency_list[u]:
                in_degree[v_neighbor] -= 1
                if in_degree[v_neighbor] == 0:
                    queue.append(v_neighbor)
        if len(self.node_order) != len(node_ids):
            raise ValueError(f"Graph has a cycle or is disconnected. Order: {self.node_order}, Degrees: {in_degree}")
        log_status(f"[GraphOrchestrator] INFO: Node execution order: {self.node_order}")

    def _initialize_agents(self):
        agent_class_map = {
            "PDFLoaderAgent": PDFLoaderAgent,
            "PDFSummarizerAgent": PDFSummarizerAgent,
            "MultiDocSynthesizerAgent": MultiDocSynthesizerAgent,
            "WebResearcherAgent": WebResearcherAgent,
            "ExperimentalDataLoaderAgent": ExperimentalDataLoaderAgent,
            "KnowledgeIntegratorAgent": KnowledgeIntegratorAgent,
            "HypothesisGeneratorAgent": HypothesisGeneratorAgent,
            "ExperimentDesignerAgent": ExperimentDesignerAgent,
        }
        for node_def in self.graph_definition.get('nodes', []):
            agent_id = node_def['id']
            agent_type_name = node_def['type']
            agent_config_params = node_def.get('config', {})
            agent_class = agent_class_map.get(agent_type_name)
            if not agent_class:
                raise ValueError(f"Unknown agent type: {agent_type_name} for node {agent_id}")
            self.agents[agent_id] = agent_class(agent_id, agent_type_name, agent_config_params)

    def run(self, all_pdf_paths: list, experimental_data_file_path: str, project_base_output_dir: str):
        outputs_history = {}
        log_status(
            f"[GraphOrchestrator] Starting INTEGRATED workflow for {len(all_pdf_paths)} PDFs. Output: {project_base_output_dir}")

        all_summaries_for_synthesis = []
        pdf_loader_agent = self.agents.get("pdf_loader_node")
        pdf_summarizer_agent = self.agents.get("pdf_summarizer_node")

        if not pdf_loader_agent or not pdf_summarizer_agent:
            log_status(
                "[GraphOrchestrator] ERROR: PDFLoaderAgent or PDFSummarizerAgent not defined in graph nodes. Cannot process PDFs.")
            return {"error": "Essential PDF processing agents not found in graph."}

        for i, pdf_path in enumerate(all_pdf_paths):
            log_status(f"\n[GraphOrchestrator] Processing PDF {i + 1}/{len(all_pdf_paths)}: {pdf_path}")
            load_input = {"pdf_path": pdf_path}
            load_output = pdf_loader_agent.execute(load_input)
            outputs_history[f"pdf_loader_{i}"] = load_output
            if load_output.get("error"):
                log_status(f"[GraphOrchestrator] ERROR loading PDF {pdf_path}: {load_output['error']}")
                all_summaries_for_synthesis.append(
                    {"summary": f"Error loading {pdf_path}: {load_output['error']}", "original_pdf_path": pdf_path,
                     "error": True})
                continue

            summarize_input = {"pdf_text_content": load_output["pdf_text_content"], "original_pdf_path": pdf_path}
            summary_output = pdf_summarizer_agent.execute(summarize_input)
            outputs_history[f"pdf_summarizer_{i}"] = summary_output
            if summary_output.get("error"):
                log_status(f"[GraphOrchestrator] ERROR summarizing PDF {pdf_path}: {summary_output['error']}")
                all_summaries_for_synthesis.append(
                    {"summary": f"Error summarizing {pdf_path}: {summary_output['error']}",
                     "original_pdf_path": pdf_path, "error": True})
            else:
                all_summaries_for_synthesis.append(
                    {"summary": summary_output["summary"], "original_pdf_path": pdf_path})

        if "multi_doc_synthesizer" in self.agents:
            mds_inputs = {"all_pdf_summaries": all_summaries_for_synthesis}
            log_status(f"\n[GraphOrchestrator] EXECUTING_NODE: 'multi_doc_synthesizer'")
            outputs_history["multi_doc_synthesizer"] = self.agents["multi_doc_synthesizer"].execute(mds_inputs)
            log_status(
                f"[multi_doc_synthesizer] RESULT: {str(outputs_history['multi_doc_synthesizer'].get('multi_doc_synthesis_output', 'N/A')[:100])}...")
            if outputs_history["multi_doc_synthesizer"].get("error"):
                log_status(
                    f"[GraphOrchestrator] NODE_EXECUTION_ERROR: multi_doc_synthesizer: {outputs_history['multi_doc_synthesizer']['error']}")

        if "experimental_data_loader" in self.agents and experimental_data_file_path:
            exp_data_inputs = {"experimental_data_file_path": experimental_data_file_path}
            log_status(f"\n[GraphOrchestrator] EXECUTING_NODE: 'experimental_data_loader'")
            outputs_history["experimental_data_loader"] = self.agents["experimental_data_loader"].execute(
                exp_data_inputs)
            log_status(
                f"[experimental_data_loader] RESULT: {str(outputs_history['experimental_data_loader'].get('experimental_data_summary', 'N/A')[:100])}...")
            if outputs_history["experimental_data_loader"].get("error"):
                log_status(
                    f"[GraphOrchestrator] NODE_EXECUTION_ERROR: experimental_data_loader: {outputs_history['experimental_data_loader']['error']}")
        elif "experimental_data_loader" in self.agents:
            outputs_history["experimental_data_loader"] = {"experimental_data_summary": "N/A - No file provided."}

        manually_processed_nodes = {"pdf_loader_node", "pdf_summarizer_node", "multi_doc_synthesizer",
                                    "experimental_data_loader"}

        for node_id in self.node_order:
            if node_id in manually_processed_nodes:
                continue

            log_status(f"\n[GraphOrchestrator] EXECUTING_NODE: '{node_id}' (Type: {self.agents[node_id].agent_type})")
            current_agent = self.agents[node_id]
            agent_inputs = {}
            for edge_def in self.graph_definition.get('edges', []):
                if edge_def['to'] == node_id:
                    from_node_id = edge_def['from']
                    if from_node_id not in outputs_history:
                        log_status(
                            f"[GraphOrchestrator] ERROR: Output from '{from_node_id}' not found for '{node_id}'.")
                        data_mapping = edge_def.get("data_mapping", {})
                        for _, target_key in data_mapping.items():
                            agent_inputs[target_key] = f"Error: Input from '{from_node_id}' missing."
                            agent_inputs[f"{target_key}_error"] = True
                        continue

                    source_outputs = outputs_history[from_node_id]
                    source_node_had_error = bool(source_outputs.get("error"))
                    data_mapping = edge_def.get("data_mapping")

                    if data_mapping:
                        for src_key, target_key in data_mapping.items():
                            if src_key in source_outputs:
                                agent_inputs[target_key] = source_outputs[src_key]
                                if source_node_had_error or (
                                        isinstance(source_outputs[src_key], str) and source_outputs[src_key].startswith(
                                        "Error:")):
                                    agent_inputs[f"{target_key}_error"] = True
                            else:
                                agent_inputs[target_key] = f"Error: Key '{src_key}' missing from '{from_node_id}'."
                                agent_inputs[f"{target_key}_error"] = True
                    else:
                        agent_inputs.update(source_outputs)
                        if source_node_had_error:
                            for key in source_outputs:
                                if key != "error": agent_inputs[f"{key}_error"] = True

            log_status(
                f"[{node_id}] INFO: Inputs gathered: {{ {', '.join([f'{k}: {str(v)[:50]}...' for k, v in agent_inputs.items()])} }}")

            node_output = {}
            try:
                if isinstance(current_agent, ExperimentDesignerAgent) and "hypotheses_list" in agent_inputs:
                    hypotheses_list_input = agent_inputs.get("hypotheses_list", [])
                    hypotheses_list_had_upstream_error = agent_inputs.get("hypotheses_list_error", False)
                    if not isinstance(hypotheses_list_input, list): hypotheses_list_input = [
                        hypotheses_list_input] if hypotheses_list_input else []
                    all_designs = []
                    for i, hypo_str in enumerate(hypotheses_list_input):
                        single_hypo_input = {"hypothesis": hypo_str}
                        if hypotheses_list_had_upstream_error or (
                                isinstance(hypo_str, str) and hypo_str.startswith("Error:")):
                            single_hypo_input["hypothesis_error"] = True
                        design_output_single = current_agent.execute(single_hypo_input)
                        all_designs.append(design_output_single)
                    node_output = {"experiment_designs_list": all_designs}
                else:
                    node_output = current_agent.execute(agent_inputs)
            except Exception as agent_exec_e:
                log_status(
                    f"[GraphOrchestrator] NODE_EXECUTION_CRITICAL_ERROR: Node '{node_id}' failed: {agent_exec_e}")
                node_output = {"error": f"Agent execution for '{node_id}' failed critically: {agent_exec_e}"}

            outputs_history[node_id] = node_output
            log_status(
                f"[{node_id}] RESULT: {{ {', '.join([f'{k}: {str(v)[:70]}...' for k, v in node_output.items()])} }}")
            if node_output.get("error"):
                log_status(
                    f"[GraphOrchestrator] NODE_EXECUTION_ERROR_REPORTED: Node '{node_id}': {node_output['error']}")

        self._save_consolidated_outputs(outputs_history, project_base_output_dir)
        log_status("\n[GraphOrchestrator] INFO: INTEGRATED workflow execution completed.")
        return outputs_history

    def _save_consolidated_outputs(self, outputs_history: dict, project_base_output_dir: str):
        log_status(f"[GraphOrchestrator] Saving consolidated outputs to {project_base_output_dir}...")
        output_dirs_config = APP_CONFIG.get("system_variables", {})
        project_output_paths = {}
        for key, subfolder_name in output_dirs_config.items():
            if key.startswith("output_project_") and key.endswith("_folder_name"):
                full_path = os.path.join(project_base_output_dir, subfolder_name)
                project_output_paths[key.replace("output_project_", "").replace("_folder_name", "")] = full_path
                if not os.path.exists(full_path):
                    try:
                        os.makedirs(full_path)
                    except OSError as e:
                        log_status(f"ERROR creating output subfolder {full_path}: {e}")

        mds_out = outputs_history.get("multi_doc_synthesizer", {}).get("multi_doc_synthesis_output")
        synthesis_path = project_output_paths.get("synthesis")
        if mds_out and not mds_out.startswith("Error:") and synthesis_path and os.path.exists(synthesis_path):
            with open(os.path.join(synthesis_path, "multi_document_synthesis.txt"), "w", encoding="utf-8") as f:
                f.write(mds_out)
            log_status(f"Saved multi-document synthesis to {synthesis_path}")

        web_research_out = outputs_history.get("web_researcher", {}).get("web_summary")
        if web_research_out and not web_research_out.startswith("Error:") and synthesis_path and os.path.exists(
                synthesis_path):
            with open(os.path.join(synthesis_path, "web_research_on_synthesis.txt"), "w", encoding="utf-8") as f:
                f.write(web_research_out)
            log_status(f"Saved web research (on synthesis) to {synthesis_path}")

        exp_data_out = outputs_history.get("experimental_data_loader", {}).get("experimental_data_summary", "N/A")
        if exp_data_out != "N/A" and not exp_data_out.startswith("Error:") and synthesis_path and os.path.exists(
                synthesis_path):
            with open(os.path.join(synthesis_path, "experimental_data_summary.txt"), "w", encoding="utf-8") as f:
                f.write(exp_data_out)
            log_status(f"Saved experimental data summary to {synthesis_path}")

        ikb_out = outputs_history.get("knowledge_integrator", {}).get("integrated_knowledge_brief")
        if ikb_out and not ikb_out.startswith("Error:") and synthesis_path and os.path.exists(synthesis_path):
            with open(os.path.join(synthesis_path, "integrated_knowledge_brief.txt"), "w", encoding="utf-8") as f:
                f.write(ikb_out)
            log_status(f"Saved integrated knowledge brief to {synthesis_path}")

        hypo_gen_node_out = outputs_history.get("hypothesis_generator", {})
        hypo_path = project_output_paths.get("hypotheses")
        if hypo_path and os.path.exists(hypo_path):
            raw_blob = hypo_gen_node_out.get("hypotheses_output_blob")
            if raw_blob and not raw_blob.startswith("Error:"):
                with open(os.path.join(hypo_path, "hypotheses_raw_llm_output.txt"), "w",
                          encoding="utf-8") as f: f.write(raw_blob)

            key_ops = hypo_gen_node_out.get("key_opportunities")
            if key_ops:
                with open(os.path.join(hypo_path, "key_research_opportunities.txt"), "w",
                          encoding="utf-8") as f: f.write(key_ops)

            hypo_list = hypo_gen_node_out.get("hypotheses_list", [])
            if hypo_list:
                with open(os.path.join(hypo_path, "hypotheses_list.txt"), "w", encoding="utf-8") as f:
                    for i, h in enumerate(hypo_list): f.write(f"{i + 1}. {h}\n\n")
            log_status(f"Saved hypothesis outputs to {hypo_path}")

        exp_designs = outputs_history.get("experiment_designer", {}).get("experiment_designs_list", [])
        exp_path = project_output_paths.get("experiments")
        if exp_designs and exp_path and os.path.exists(exp_path):
            for i, design_info in enumerate(exp_designs):
                hypo = design_info.get("hypothesis_processed", f"N/A_{i + 1}")
                design = design_info.get("experiment_design", "")
                err = design_info.get("error")
                safe_hypo = "".join(c if c.isalnum() else "_" for c in str(hypo)[:30]).strip('_') or f"hypo_{i + 1}"
                fname = f"exp_design_{i + 1}_{safe_hypo}.txt"
                with open(os.path.join(exp_path, fname), "w", encoding="utf-8") as f:
                    f.write(f"Hypothesis: {hypo}\n\n")
                    if err: f.write(f"Error in design: {err}\n")
                    f.write(f"Experiment Design:\n{design}\n")
            log_status(f"Saved {len(exp_designs)} experiment designs to {exp_path}")


# --- Main Orchestration Function (Callable by GUI) ---
def run_project_orchestration(
        pdf_file_paths: list,
        experimental_data_path: str,
        project_base_output_dir: str,
        status_update_callback: callable,
        config_file_path: str = "config.json"):
    set_status_callback(status_update_callback)
    resolved_config_path = config_file_path
    if not os.path.isabs(config_file_path):
        resolved_config_path = os.path.join(SCRIPT_DIR, config_file_path)

    if not load_app_config(resolved_config_path):
        return {"error": f"Config load failed from '{resolved_config_path}'."}
    if not APP_CONFIG:
        return {"error": "APP_CONFIG empty after load attempt."}

    openai_api_key = APP_CONFIG.get("system_variables", {}).get("openai_api_key")
    if not openai_api_key or openai_api_key in ["YOUR_OPENAI_API_KEY_NOT_IN_CONFIG", "YOUR_ACTUAL_OPENAI_API_KEY",
                                                "KEY"]:
        log_status(f"[MainWorkflow] CONFIG ERROR: OpenAI API key missing/placeholder in '{resolved_config_path}'.")

    if not pdf_file_paths:
        log_status("[MainWorkflow] ERROR: No PDF files provided for processing.")
        return {"error": "No PDF files provided."}
    for p_path in pdf_file_paths:
        if not os.path.exists(p_path):
            log_status(f"[MainWorkflow] ERROR: Input PDF not found at '{p_path}'.")
            return {"error": f"Input PDF not found: {p_path}"}

    if not os.path.exists(project_base_output_dir):
        try:
            os.makedirs(project_base_output_dir, exist_ok=True)
            log_status(f"[MainWorkflow] Created project output directory: {project_base_output_dir}")
        except OSError as e:
            log_status(f"[MainWorkflow] ERROR creating project output directory '{project_base_output_dir}': {e}")
            return {"error": f"Could not create project output directory: {e}"}

    try:
        orchestrator = GraphOrchestrator(APP_CONFIG.get("graph_definition"))
        final_outputs = orchestrator.run(
            all_pdf_paths=pdf_file_paths,
            experimental_data_file_path=experimental_data_path,
            project_base_output_dir=project_base_output_dir
        )
        log_status("[MainWorkflow] INTEGRATED project orchestration finished.")
        return final_outputs
    except Exception as e:
        log_status(f"[MainWorkflow] ERROR: Orchestration failed: {e}")
        import traceback
        log_status(traceback.format_exc())
        return {"error": f"Unexpected error in orchestration: {e}"}


# --- Main Execution (for standalone testing) ---
if __name__ == "__main__":
    CLI_CONFIG_FILENAME = "config_cli_test_integrated.json"
    cli_config_full_path_check = os.path.join(SCRIPT_DIR, CLI_CONFIG_FILENAME)

    if not os.path.exists(cli_config_full_path_check):
        print(f"[CLI_Test] '{CLI_CONFIG_FILENAME}' not found. Creating a minimal one for integrated test.")
        dummy_cli_config = json.loads("""
        {
          "system_variables": {
            "openai_api_key": "YOUR_OPENAI_API_KEY_IN_CLI_CONFIG",
            "output_project_synthesis_folder_name": "project_synthesis_cli",
            "output_project_hypotheses_folder_name": "project_hypotheses_cli",
            "output_project_experiments_folder_name": "project_experiments_cli",
            "default_llm_model": "gpt-3.5-turbo",
            "models": {
              "pdf_summarizer": "gpt-3.5-turbo",
              "multi_doc_synthesizer_model": "gpt-3.5-turbo",
              "web_research_model": "gpt-3.5-turbo",
              "knowledge_integrator_model": "gpt-3.5-turbo",
              "hypothesis_generator": "gpt-3.5-turbo",
              "experiment_designer": "gpt-3.5-turbo"
            },
            "openai_api_timeout_seconds": 120
          },
          "agent_prompts": {
            "pdf_summarizer_sm": "Summarize this text in one concise paragraph: {text_content}",
            "multi_doc_synthesizer_sm": "Synthesize these summaries: {all_summaries_text}. Output: Cross-document understanding.",
            "web_researcher_sm": "Based on: {cross_document_understanding}, find web info and synthesize. If no live search, say so.",
            "experimental_data_loader_sm": "This is experimental data: {data_content}. Summarize if complex, else pass as is.",
            "knowledge_integrator_sm": "Integrate: Multi-doc: {multi_doc_synthesis}, Web: {web_research_summary}, ExpData: {experimental_data_summary}. Output: Integrated brief.",
            "hypothesis_generator_sm": "From: {integrated_knowledge_brief}. Output JSON: {\\"key_opportunities\\": \\"ops...\\", \\"hypotheses\\": [\\"hypo1...\\"]}",
            "experiment_designer_sm": "Design experiment for: {hypothesis}"
          },
          "graph_definition": {
            "nodes": [
              {"id": "pdf_loader_node", "type": "PDFLoaderAgent", "config": {}},
              {"id": "pdf_summarizer_node", "type": "PDFSummarizerAgent", "config": {"model_key": "pdf_summarizer", "system_message_key": "pdf_summarizer_sm"}},
              {"id": "multi_doc_synthesizer", "type": "MultiDocSynthesizerAgent", "config": {"model_key": "multi_doc_synthesizer_model", "system_message_key": "multi_doc_synthesizer_sm"}},
              {"id": "web_researcher", "type": "WebResearcherAgent", "config": {"model_key": "web_research_model", "system_message_key": "web_researcher_sm"}},
              {"id": "experimental_data_loader", "type": "ExperimentalDataLoaderAgent", "config": {"system_message_key": "experimental_data_loader_sm"}},
              {"id": "knowledge_integrator", "type": "KnowledgeIntegratorAgent", "config": {"model_key": "knowledge_integrator_model", "system_message_key": "knowledge_integrator_sm"}},
              {"id": "hypothesis_generator", "type": "HypothesisGeneratorAgent", "config": {"model_key": "hypothesis_generator", "system_message_key": "hypothesis_generator_sm"}},
              {"id": "experiment_designer", "type": "ExperimentDesignerAgent", "config": {"model_key": "experiment_designer", "system_message_key": "experiment_designer_sm"}}
            ],
            "edges": [
              {"from": "multi_doc_synthesizer", "to": "web_researcher", "data_mapping": {"multi_doc_synthesis_output": "cross_document_understanding"}},
              {"from": "multi_doc_synthesizer", "to": "knowledge_integrator", "data_mapping": {"multi_doc_synthesis_output": "multi_doc_synthesis"}},
              {"from": "web_researcher", "to": "knowledge_integrator", "data_mapping": {"web_summary": "web_research_summary"}},
              {"from": "experimental_data_loader", "to": "knowledge_integrator", "data_mapping": {"experimental_data_summary": "experimental_data_summary"}},
              {"from": "knowledge_integrator", "to": "hypothesis_generator", "data_mapping": {"integrated_knowledge_brief": "integrated_knowledge_brief"}},
              {"from": "hypothesis_generator", "to": "experiment_designer", "data_mapping": {"hypotheses_list": "hypotheses_list"}}
            ]
          }
        }
        """)
        try:
            with open(cli_config_full_path_check, 'w') as f:
                json.dump(dummy_cli_config, f, indent=2)
            print(f"[CLI_Test] Created '{cli_config_full_path_check}'. Please review and add API keys.")
        except Exception as e:
            print(f"[CLI_Test] ERROR writing dummy CLI config: {e}")
            CLI_CONFIG_FILENAME = "config.json"
            print(f"[CLI_Test] Falling back to main 'config.json'.")

    set_status_callback(print)
    if not load_app_config(config_path=CLI_CONFIG_FILENAME):
        print(f"[CLI_Test] CRITICAL: Failed to load config '{CLI_CONFIG_FILENAME}'.")
        exit(1)

    cli_test_input_dir = os.path.join(os.getcwd(), "cli_test_multi_input_pdfs")
    cli_test_exp_data_dir = os.path.join(os.getcwd(), "cli_test_experimental_data")
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    cli_test_project_output_dir = os.path.join(os.getcwd(), f"cli_test_integrated_project_output_{timestamp}")

    for p in [cli_test_input_dir, cli_test_exp_data_dir, cli_test_project_output_dir]:
        if not os.path.exists(p): os.makedirs(p)

    pdf_paths_for_test = []
    for i in range(2):
        pdf_name = f"dummy_paper_{i + 1}.pdf"
        full_pdf_path = os.path.join(cli_test_input_dir, pdf_name)
        if REPORTLAB_AVAILABLE:
            try:
                c = canvas.Canvas(full_pdf_path, pagesize=letter)
                t = c.beginText(inch, 10 * inch)
                t.setFont("Helvetica", 12)
                t.textLine(f"CLI Test: Dummy PDF Document {i + 1}.")
                t.textLine(f"Content for paper {i + 1} focusing on topic variant {chr(65 + i)}.")
                if i == 0:
                    t.textLine("This one mentions renewable energy.")
                else:
                    t.textLine("This one discusses energy storage solutions.")
                c.drawText(t)
                c.save()
                pdf_paths_for_test.append(full_pdf_path)
                print(f"[CLI_Test] Created dummy PDF: {full_pdf_path}")
            except Exception as e:
                print(f"Error creating dummy PDF {pdf_name}: {e}")
        elif os.path.exists(full_pdf_path):
            pdf_paths_for_test.append(full_pdf_path)
            print(f"[CLI_Test] Using existing dummy PDF: {full_pdf_path}")

    if not pdf_paths_for_test:
        print(
            "[CLI_Test] No dummy PDFs created or found. Please create them manually or ensure reportlab is available.")
        exit(1)

    exp_data_filename = "dummy_experimental_results.txt"
    exp_data_full_path = os.path.join(cli_test_exp_data_dir, exp_data_filename)
    try:
        with open(exp_data_full_path, "w") as f:
            f.write("Experimental Results Summary:\n")
            f.write("Test A showed a 15% improvement in efficiency with the new catalyst.\n")
            f.write("Test B, under high-temperature conditions, indicated material degradation after 100 cycles.\n")
            f.write("Further research into material stability is recommended.")
        print(f"[CLI_Test] Created dummy experimental data: {exp_data_full_path}")
    except Exception as e:
        print(f"Error creating dummy experimental data: {e}")
        exp_data_full_path = None

    print(f"\n[CLI_Test] Running INTEGRATED project orchestration for {len(pdf_paths_for_test)} PDFs.")
    print(f"[CLI_Test] Experimental data: {exp_data_full_path if exp_data_full_path else 'N/A'}")
    print(f"[CLI_Test] Project outputs will be in: {cli_test_project_output_dir}")
    print(f"[CLI_Test] Using config: {os.path.join(SCRIPT_DIR, CLI_CONFIG_FILENAME)}\n")

    results = run_project_orchestration(
        pdf_file_paths=pdf_paths_for_test,
        experimental_data_path=exp_data_full_path,
        project_base_output_dir=cli_test_project_output_dir,
        status_update_callback=print,
        config_file_path=CLI_CONFIG_FILENAME
    )

    print("\n" + "=" * 30 + " CLI INTEGRATED TEST FINAL RESULTS " + "=" * 30)
    if results.get("error"):
        print(f"CLI Test Run completed with an error: {results['error']}")
    else:
        print(f"Full results structure logged. Key outputs will be in '{cli_test_project_output_dir}'.")
        hypo_gen_output = results.get("hypothesis_generator", {})
        hypotheses = hypo_gen_output.get("hypotheses_list", [])
        if hypotheses:
            print(f"First generated hypothesis: {hypotheses[0][:200]}...")
        else:
            print("No hypotheses were generated or parsed.")
    print("=" * (60 + len(" CLI INTEGRATED TEST FINAL RESULTS ")))
    print("\n--- Multi-Agent LLM System Backend CLI Integrated Test Finished ---")

