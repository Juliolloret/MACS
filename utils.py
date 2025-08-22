import os
import json
from typing import Optional

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
    Returns the config dictionary on success, None on failure.
    """
    global PyPDF2, REPORTLAB_AVAILABLE, canvas, letter, inch, openai_errors, OPENAI_SDK_AVAILABLE, set_default_openai_key

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
            config = json.load(f)
        log_status(f"[AppConfig] Successfully loaded configuration from '{resolved_config_path}'.")

        # Dynamic library loading remains as a side-effect for now
        try:
            import PyPDF2 as PyPDF2_lib
            PyPDF2 = PyPDF2_lib
        except ImportError:
            PyPDF2 = None
        try:
            from reportlab.pdfgen import canvas as rl_canvas
            from reportlab.lib.pagesizes import letter as rl_letter
            from reportlab.lib.units import inch as rl_inch
            canvas, letter, inch = rl_canvas, rl_letter, rl_inch
            REPORTLAB_AVAILABLE = True
        except ImportError:
            canvas, letter, inch = None, None, None
            REPORTLAB_AVAILABLE = False

        return config
    except FileNotFoundError:
        log_status(f"[AppConfig] ERROR: Configuration file '{resolved_config_path}' not found.")
    except json.JSONDecodeError as e:
        log_status(f"[AppConfig] ERROR: Could not decode JSON from '{resolved_config_path}': {e}.")
    except Exception as e:
        log_status(f"[AppConfig] ERROR: An unexpected error occurred while loading config '{resolved_config_path}': {e}.")
    return None


def get_model_name(app_config: dict, model_key: Optional[str] = None) -> str:
    """Retrieves a model name from the provided config, falling back to default if not found."""
    if not app_config:
        return "gpt-4o"
    models_config = app_config.get("system_variables", {}).get("models", {})
    if model_key and model_key in models_config:
        return models_config[model_key]
    return app_config.get("system_variables", {}).get("default_llm_model", "gpt-4o")


def get_prompt_text(app_config: dict, prompt_key: Optional[str]) -> str:
    """Retrieves a prompt text from the provided config by its key."""
    if prompt_key is None:
        return ""
    if not app_config:
        log_status(f"[Utils] ERROR: get_prompt_text called for '{prompt_key}' but app_config is not provided.")
        return f"ERROR: Config not provided, prompt key '{prompt_key}' unavailable."

    prompts_config = app_config.get("agent_prompts", {})
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
    """Deprecated wrapper around OpenAILLM for backward compatibility."""
    try:
        from llm_openai import OpenAILLM
    except ImportError:
        return "Error: OpenAI library is not available."

    api_key = APP_CONFIG.get("system_variables", {}).get("openai_api_key")
    timeout = float(APP_CONFIG.get("system_variables", {}).get("openai_api_timeout_seconds", 120))
    llm = OpenAILLM(api_key=api_key, timeout=int(timeout))
    return llm.complete(system=system_message, prompt=prompt,
                        model=model_name, temperature=temperature)
