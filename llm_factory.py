"""Centralized LLM client factory for consistent client creation."""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

from llm import LLMClient, LLMError


def create_llm_client(
    app_config: Dict[str, Any],
    api_key: Optional[str] = None,
    timeout: Optional[int] = None,
) -> LLMClient:
    """Create an LLM client based on application configuration.

    This factory consolidates LLM client creation logic that was previously
    duplicated across the orchestrator, adaptive runner, and GUI.

    Parameters
    ----------
    app_config : Dict[str, Any]
        Application configuration containing system_variables.
    api_key : Optional[str]
        Override API key. If not provided, uses config or environment variable.
    timeout : Optional[int]
        Override timeout in seconds. If not provided, uses config value.

    Returns
    -------
    LLMClient
        Configured LLM client instance.

    Raises
    ------
    LLMError
        If the requested LLM client type is not available or misconfigured.
    ValueError
        If an unknown client type is specified.
    """
    system_vars = app_config.get("system_variables", {})

    # Determine client type
    client_type = system_vars.get("llm_client", "openai").lower()

    # Resolve API key: parameter > config > environment
    resolved_api_key = api_key
    if resolved_api_key is None:
        resolved_api_key = system_vars.get("openai_api_key")
    if resolved_api_key is None:
        resolved_api_key = os.environ.get("OPENAI_API_KEY")

    # Resolve timeout
    resolved_timeout = timeout
    if resolved_timeout is None:
        resolved_timeout = int(system_vars.get("openai_api_timeout_seconds", 120))

    if client_type == "openai":
        return _create_openai_client(app_config, resolved_api_key, resolved_timeout)
    elif client_type == "fake":
        return _create_fake_client(app_config)
    else:
        raise ValueError(f"Unknown LLM client type: {client_type}")


def _create_openai_client(
    app_config: Dict[str, Any],
    api_key: Optional[str],
    timeout: int,
) -> LLMClient:
    """Create an OpenAI-based LLM client.

    Parameters
    ----------
    app_config : Dict[str, Any]
        Application configuration.
    api_key : Optional[str]
        OpenAI API key.
    timeout : int
        Request timeout in seconds.

    Returns
    -------
    LLMClient
        OpenAI LLM client instance.

    Raises
    ------
    LLMError
        If the OpenAI library is not available.
    """
    try:
        from llm_openai import OpenAILLM  # pylint: disable=import-outside-toplevel
    except ImportError as exc:
        raise LLMError(
            "OpenAI LLM client requested but openai library is not available. "
            "Install with: pip install openai"
        ) from exc

    return OpenAILLM(app_config=app_config, api_key=api_key, timeout=timeout)


def _create_fake_client(app_config: Dict[str, Any]) -> LLMClient:
    """Create a fake LLM client for testing.

    Parameters
    ----------
    app_config : Dict[str, Any]
        Application configuration.

    Returns
    -------
    LLMClient
        Fake LLM client instance.

    Raises
    ------
    LLMError
        If the fake client module is not available.
    """
    try:
        from llm_fake import FakeLLM  # pylint: disable=import-outside-toplevel
    except ImportError as exc:
        raise LLMError(
            "Fake LLM client requested but llm_fake module is not available."
        ) from exc

    return FakeLLM(app_config=app_config)


def get_default_model(app_config: Dict[str, Any], model_key: Optional[str] = None) -> str:
    """Get model name from config with fallback to default.

    Parameters
    ----------
    app_config : Dict[str, Any]
        Application configuration.
    model_key : Optional[str]
        Specific model key to look up in models config.

    Returns
    -------
    str
        Model name to use.
    """
    if not app_config:
        return "gpt-4o"

    system_vars = app_config.get("system_variables", {})
    models_config = system_vars.get("models", {})

    if model_key and model_key in models_config:
        return models_config[model_key]

    return system_vars.get("default_llm_model", "gpt-4o")
