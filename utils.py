"""Utility helpers for configuration management and OpenAI interactions."""

from __future__ import annotations

import json
import os
from typing import Optional

try:
    from pydantic import ValidationError
except ImportError:  # pragma: no cover - pydantic is optional at runtime
    ValidationError = Exception  # type: ignore

from config_schema import validate_graph_definition
from llm import LLMError

try:  # Expose PyPDF2 for agents that rely on it without importing here
    import PyPDF2  # pylint: disable=unused-import
except ImportError:  # pragma: no cover
    PyPDF2 = None  # type: ignore

# Attempt to import OpenAI library and its specific errors. These are primarily
# used by ``call_openai_api``.
try:  # pragma: no cover - the SDK may not be installed in test environments
    from openai import (
        APIConnectionError,
        APITimeoutError,
        AuthenticationError,
        BadRequestError,
        RateLimitError,
    )
    OPENAI_SDK_AVAILABLE = True
except ImportError:  # pragma: no cover - handled gracefully at runtime
    OPENAI_SDK_AVAILABLE = False
    APIConnectionError = APITimeoutError = RateLimitError = AuthenticationError = BadRequestError = (  # type: ignore
        Exception
    )

openai_errors = {
    "APIConnectionError": APIConnectionError,
    "APITimeoutError": APITimeoutError,
    "RateLimitError": RateLimitError,
    "AuthenticationError": AuthenticationError,
    "BadRequestError": BadRequestError,
} if OPENAI_SDK_AVAILABLE else {}


APP_CONFIG: dict = {}
_STATUS_CALLBACK = {"func": print}
_GRAPH_CALLBACK = {"func": None}

UTIL_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def set_status_callback(callback_func):
    """Set the status callback used by :func:`log_status`."""

    _STATUS_CALLBACK["func"] = callback_func


def set_graph_callback(callback_func):
    """Set the callback used to report generated graph visualizations."""

    _GRAPH_CALLBACK["func"] = callback_func


def log_status(message: str):
    """Log ``message`` using the configured status callback."""

    callback = _STATUS_CALLBACK.get("func", print)
    if callable(callback):
        callback(message)
    else:
        print(message)


def report_graph_visualization(path: str):
    """Invoke the graph visualization callback with ``path`` if set."""

    callback = _GRAPH_CALLBACK.get("func")
    if callable(callback):
        callback(path)


def load_app_config(config_path: str = "config.json", main_script_dir: Optional[str] = None):
    """Load and validate the application configuration.

    The configuration is stored in the module-level ``APP_CONFIG`` dictionary.
    Returns the loaded configuration on success, otherwise ``None``.
    """

    if main_script_dir and not os.path.isabs(config_path):
        resolved_config_path = os.path.join(main_script_dir, config_path)
    else:
        resolved_config_path = config_path

    log_status(
        f"[AppConfig] Attempting to load configuration from: '{resolved_config_path}'"
    )
    try:
        with open(resolved_config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        log_status(
            f"[AppConfig] Successfully loaded configuration from '{resolved_config_path}'."
        )

        if config.get("graph_definition"):
            validate_graph_definition(config["graph_definition"])

        APP_CONFIG.clear()
        APP_CONFIG.update(config)

        return config
    except FileNotFoundError:
        log_status(
            f"[AppConfig] ERROR: Configuration file '{resolved_config_path}' not found."
        )
    except json.JSONDecodeError as exc:
        log_status(
            f"[AppConfig] ERROR: Could not decode JSON from '{resolved_config_path}': {exc}."
        )
    except ValidationError as exc:
        log_status(
            f"[AppConfig] ERROR: Graph configuration validation failed: {exc}"
        )
    except OSError as exc:
        log_status(
            f"[AppConfig] ERROR: OS error occurred while loading config '{resolved_config_path}': {exc}."
        )
    return None


def get_model_name(app_config: dict, model_key: Optional[str] = None) -> str:
    """Retrieve a model name from the provided config, with a sensible default."""

    if not app_config:
        return "gpt-4o"
    models_config = app_config.get("system_variables", {}).get("models", {})
    if model_key and model_key in models_config:
        return models_config[model_key]
    return app_config.get("system_variables", {}).get("default_llm_model", "gpt-4o")


def get_prompt_text(app_config: dict, prompt_key: Optional[str]) -> str:
    """Retrieve prompt text from ``app_config`` using ``prompt_key``."""

    if prompt_key is None:
        return ""
    if not app_config:
        log_status(
            f"[Utils] ERROR: get_prompt_text called for '{prompt_key}' but app_config is not provided."
        )
        return f"ERROR: Config not provided, prompt key '{prompt_key}' unavailable."

    prompts_config = app_config.get("agent_prompts", {})
    if prompt_key not in prompts_config:
        log_status(
            f"[AppConfig] ERROR: Prompt key '{prompt_key}' not found in agent_prompts."
        )
        return f"ERROR: Prompt key '{prompt_key}' not found."

    prompt_text = prompts_config.get(prompt_key)
    if prompt_text is None:
        log_status(
            f"[AppConfig] WARNING: Prompt key '{prompt_key}' has null value in config. Returning empty string."
        )
        return ""
    return prompt_text


def call_openai_api(
    prompt: str,
    system_message: str = "You are a helpful assistant.",
    _agent_name: str = "LLM",
    model_name: Optional[str] = None,
    temperature: float = 0.5,
) -> str:
    """Deprecated wrapper around :class:`llm_openai.OpenAILLM` for backward compatibility."""

    try:
        from llm_openai import OpenAILLM  # pylint: disable=import-outside-toplevel
    except ImportError as exc:  # pragma: no cover - handled in tests
        raise LLMError("OpenAI library is not available.") from exc

    api_key = APP_CONFIG.get("system_variables", {}).get("openai_api_key")
    timeout = float(
        APP_CONFIG.get("system_variables", {}).get("openai_api_timeout_seconds", 120)
    )
    llm = OpenAILLM(app_config=APP_CONFIG, api_key=api_key, timeout=int(timeout))
    return llm.complete(
        system=system_message,
        prompt=prompt,
        model=model_name,
        temperature=temperature,
    )

