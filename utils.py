import os
import json
from typing import Optional, Dict, Any

# Attempt to import OpenAI library and its specific errors.
# These are primarily used by call_openai_api.
try:
    from openai import OpenAI as OpenAI_lib, APIConnectionError, APITimeoutError, RateLimitError, \
        AuthenticationError, BadRequestError
    OPENAI_SDK_AVAILABLE = True
except ImportError:
    OPENAI_SDK_AVAILABLE = False
    OpenAI_lib = None # Placeholder if not available
    APIConnectionError = APITimeoutError = RateLimitError = AuthenticationError = BadRequestError = Exception # Fallback to base Exception

# --- Global Variables / Placeholders ---
# These will be initialized or updated by load_app_config
OpenAI = None
PyPDF2 = None
REPORTLAB_AVAILABLE = False
canvas, letter, inch = None, None, None
openai_errors = {
    "APIConnectionError": APIConnectionError,
    "APITimeoutError": APITimeoutError,
    "RateLimitError": RateLimitError,
    "AuthenticationError": AuthenticationError,
    "BadRequestError": BadRequestError
} if OPENAI_SDK_AVAILABLE else {}


APP_CONFIG = {}
STATUS_CALLBACK = print

UTIL_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# --- SDK Imports & Availability Check (Moved from multi_agent_llm_system.py) ---
# NOTE: 'openai_agents' is a placeholder for the actual SDK package name.
# This needs to be verified and updated if the SDK is published under a different name.
SDK_AVAILABLE = False
SDSAgent, Runner, WebSearchTool, ModelSettings = None, None, None, None
_sdk_import_error = None # This will store the import error if any

# Define set_default_openai_key as a placeholder if not imported,
# or ensure it's imported if it's truly from the SDK.
# For now, assuming it comes with the SDK components.
set_default_openai_key = None

try:
    # Attempt to import from a hypothetical specific SDK package name
    from openai_agents import (
        Agent as SDSAgent_actual,
        Runner as Runner_actual,
        WebSearchTool as WebSearchTool_actual,
        ModelSettings as ModelSettings_actual,
        set_default_openai_key as set_default_openai_key_actual
    )

    SDSAgent, Runner, WebSearchTool, ModelSettings = (
        SDSAgent_actual, Runner_actual, WebSearchTool_actual, ModelSettings_actual
    )
    set_default_openai_key = set_default_openai_key_actual
    SDK_AVAILABLE = True
except ImportError as e:
    _sdk_import_error = e
    # Globals SDSAgent etc. remain None. SDK_AVAILABLE remains False.
    # set_default_openai_key remains None or its placeholder value.
    pass
# --- End SDK Imports & Availability Check ---


# --- Utility Function Definitions ---

def set_status_callback(callback_func):
    """Sets a global callback for status updates."""
    global STATUS_CALLBACK
    STATUS_CALLBACK = callback_func

def log_status(message: str):
    """Logs a message using the globally defined status callback."""
    if callable(STATUS_CALLBACK):
        STATUS_CALLBACK(message)
    else:
        print(message) # Fallback to print if callback is not callable

# SDK Availability Logging (using log_status from this file)
# This needs to be called *after* log_status is defined.
if SDK_AVAILABLE:
    log_status("INFO: (from utils.py) openai-agents SDK loaded successfully.")
else:
    error_message_suffix = f" (Error: {_sdk_import_error})" if _sdk_import_error else ""
    log_status(
        f"WARNING: (from utils.py) openai-agents SDK not found or failed to import{error_message_suffix}. "
        "Ensure 'openai-agents' package is installed."
    )


def load_app_config(config_path="config.json", main_script_dir=None):
    """
    Loads the application configuration from a JSON file.
    Dynamically imports PyPDF2 and reportlab if available.
    Uses main_script_dir to resolve relative config_path if provided,
    otherwise assumes config_path is absolute or relative to where load_app_config is called.
    """
    global APP_CONFIG, OpenAI, PyPDF2, REPORTLAB_AVAILABLE, canvas, letter, inch, openai_errors, OPENAI_SDK_AVAILABLE, set_default_openai_key

    OpenAI = None
    PyPDF2 = None
    REPORTLAB_AVAILABLE = False
    canvas, letter, inch = None, None, None

    if main_script_dir and not os.path.isabs(config_path):
        resolved_config_path = os.path.join(main_script_dir, config_path)
    else:
        resolved_config_path = config_path

    log_status(f"[AppConfig] Attempting to load configuration from: '{resolved_config_path}'")
    try:
        with open(resolved_config_path, 'r', encoding='utf-8') as f:
            APP_CONFIG = json.load(f)
        log_status(f"[AppConfig] Successfully loaded configuration from '{resolved_config_path}'.")

        if OPENAI_SDK_AVAILABLE:
            log_status("[AppConfig] OpenAI library (openai>=1.0.0) confirmed available.")
        else:
            log_status("[AppConfig] WARNING: OpenAI library (openai>=1.0.0) not found. `call_openai_api` will fail.")

        try:
            import PyPDF2 as PyPDF2_lib
            PyPDF2 = PyPDF2_lib
            log_status("[AppConfig] PyPDF2 library loaded.")
        except ImportError:
            PyPDF2 = None
            log_status("[AppConfig] WARNING: PyPDF2 library not found (pip install PyPDF2).")
        try:
            from reportlab.pdfgen import canvas as rl_canvas
            from reportlab.lib.pagesizes import letter as rl_letter
            from reportlab.lib.units import inch as rl_inch
            canvas, letter, inch = rl_canvas, rl_letter, rl_inch
            REPORTLAB_AVAILABLE = True
            log_status("[AppConfig] reportlab library loaded.")
        except ImportError:
            canvas, letter, inch = None, None, None
            REPORTLAB_AVAILABLE = False
            log_status("[AppConfig] WARNING: reportlab library not found. PDF output features unavailable.")

        # Set OpenAI API key for the SDK globally, if SDK is available and key is present
        # This is done *after* APP_CONFIG is loaded.
        if SDK_AVAILABLE and callable(set_default_openai_key):
            sdk_api_key = APP_CONFIG.get("system_variables", {}).get("openai_api_key")
            if sdk_api_key and sdk_api_key not in ["YOUR_OPENAI_API_KEY_NOT_IN_CONFIG", "YOUR_ACTUAL_OPENAI_API_KEY", "KEY"]:
                try:
                    set_default_openai_key(sdk_api_key)
                    log_status("[AppConfig] OpenAI API key set for 'openai_agents' SDK via set_default_openai_key.")
                except Exception as e:
                    log_status(f"[AppConfig] WARNING: Failed to set OpenAI API key for 'openai_agents' SDK: {e}")
            else:
                log_status("[AppConfig] WARNING: Valid OpenAI API key not found in APP_CONFIG to set for 'openai_agents' SDK. SDK calls might fail if OPENAI_API_KEY env var is not set.")
        elif SDK_AVAILABLE:
            log_status("[AppConfig] WARNING: 'set_default_openai_key' function not available from 'openai_agents' SDK import. SDK calls might fail if OPENAI_API_KEY env var is not set.")

        return True
    except FileNotFoundError:
        log_status(f"[AppConfig] ERROR: Configuration file '{resolved_config_path}' not found.")
    except json.JSONDecodeError as e:
        log_status(f"[AppConfig] ERROR: Could not decode JSON from '{resolved_config_path}': {e}.")
    except Exception as e:
        log_status(f"[AppConfig] ERROR: An unexpected error occurred while loading config '{resolved_config_path}': {e}.")
    APP_CONFIG = {}
    return False


def get_model_name(model_key: Optional[str] = None) -> str:
    """Retrieves a model name from APP_CONFIG, falling back to default if not found."""
    if not APP_CONFIG:
        return "gpt-4o"
    models_config = APP_CONFIG.get("system_variables", {}).get("models", {})
    if model_key and model_key in models_config:
        return models_config[model_key]
    return APP_CONFIG.get("system_variables", {}).get("default_llm_model", "gpt-4o")


def get_prompt_text(prompt_key: Optional[str]) -> str:
    """Retrieves a prompt text from APP_CONFIG by its key."""
    if prompt_key is None:
        return ""
    if not APP_CONFIG:
        log_status(f"[Utils] ERROR: get_prompt_text called for '{prompt_key}' before APP_CONFIG initialized.")
        return f"ERROR: Config not loaded, prompt key '{prompt_key}' unavailable."

    prompts_config = APP_CONFIG.get("agent_prompts", {})
    if prompt_key not in prompts_config:
        log_status(f"[AppConfig] ERROR: Prompt key '{prompt_key}' not found in agent_prompts.")
        return f"ERROR: Prompt key '{prompt_key}' not found."

    prompt_text = prompts_config.get(prompt_key)
    if prompt_text is None:
        log_status(f"[AppConfig] WARNING: Prompt key '{prompt_key}' has null value in config. Returning empty string.")
        return ""
    return prompt_text


def call_openai_api(prompt: str, system_message: str = "You are a helpful assistant.",
                    agent_name: str = "LLM", model_name: Optional[str] = None,
                    temperature: float = 0.5) -> str:
    """Makes a call to the OpenAI ChatCompletion API."""
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

    if not OPENAI_SDK_AVAILABLE or not OpenAI_lib:
        return f"Error: OpenAI library (openai>=1.0.0) not available for model {chosen_model}."
    if not APP_CONFIG:
        return f"Error: Application configuration not loaded for model {chosen_model}."

    api_key_to_use = APP_CONFIG.get("system_variables", {}).get("openai_api_key")
    api_timeout_seconds = float(APP_CONFIG.get("system_variables", {}).get("openai_api_timeout_seconds", 120))

    if not api_key_to_use or api_key_to_use in ["YOUR_OPENAI_API_KEY_NOT_IN_CONFIG", "YOUR_ACTUAL_OPENAI_API_KEY", "KEY"]:
        return f"Error: OpenAI API key not configured for model {chosen_model}."

    try:
        client = OpenAI_lib(api_key=api_key_to_use, timeout=api_timeout_seconds)

        api_call_params = {
            "model": chosen_model,
            "messages": [
                {"role": "system", "content": effective_system_message},
                {"role": "user", "content": prompt}
            ],
            "temperature": current_temperature
        }
        response = client.chat.completions.create(**api_call_params)

        if not response.choices:
            log_status(f"[{agent_name}] LLM_CALL_ERROR: Model='{chosen_model}' response has no choices.")
            return f"Error: OpenAI API response had no choices for model {chosen_model}."

        first_choice = response.choices[0]
        if not first_choice.message or first_choice.message.content is None:
            log_status(f"[{agent_name}] LLM_CALL_SUCCESS_EMPTY_CONTENT: Model='{chosen_model}' returned None or no message content.")
            return ""

        raw_content = first_choice.message.content
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
            for err_name, err_class_obj in openai_errors.items():
                if isinstance(e, err_class_obj):
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
                        except Exception: pass
                    return f"Error: OpenAI API {err_name} for {chosen_model}: {error_detail}"

        log_status(f"[{agent_name}] LLM_ERROR (General {error_type_name}): API call with {chosen_model} failed: {e}")
        return f"Error: API call with {chosen_model} failed ({error_type_name}): {e}"
