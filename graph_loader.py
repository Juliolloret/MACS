"""Graph configuration loading utilities with environment variable substitution."""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, Optional

from config_schema import validate_graph_definition


# Pattern to match ${VAR_NAME} or ${VAR_NAME:-default} syntax
ENV_VAR_PATTERN = re.compile(r"\$\{([^}:]+)(?::-([^}]*))?\}")


def substitute_env_vars(value: Any) -> Any:
    """Recursively substitute environment variables in configuration values.

    Supports syntax:
    - ${VAR_NAME} - substitutes with env var value, raises if not set
    - ${VAR_NAME:-default} - substitutes with default if env var not set

    Parameters
    ----------
    value : Any
        The configuration value to process. Can be a string, dict, list, or other.

    Returns
    -------
    Any
        The value with environment variables substituted.

    Raises
    ------
    ValueError
        If an environment variable is referenced but not set and no default provided.
    """
    if isinstance(value, str):
        return _substitute_string(value)
    elif isinstance(value, dict):
        return {k: substitute_env_vars(v) for k, v in value.items()}
    elif isinstance(value, list):
        return [substitute_env_vars(item) for item in value]
    return value


def _substitute_string(text: str) -> str:
    """Substitute environment variables in a single string."""

    def replacer(match: re.Match) -> str:
        var_name = match.group(1)
        default_value = match.group(2)

        env_value = os.environ.get(var_name)
        if env_value is not None:
            return env_value
        if default_value is not None:
            return default_value
        raise ValueError(
            f"Environment variable '{var_name}' is not set and no default provided. "
            f"Set the variable or use ${{VAR_NAME:-default}} syntax."
        )

    return ENV_VAR_PATTERN.sub(replacer, text)


def load_config(
    config_path: str,
    base_dir: Optional[str] = None,
    substitute_env: bool = True,
) -> Dict[str, Any]:
    """Load and validate a configuration file.

    Parameters
    ----------
    config_path : str
        Path to the JSON configuration file.
    base_dir : Optional[str]
        Base directory to resolve relative paths. If provided and config_path
        is relative, the path is resolved relative to base_dir.
    substitute_env : bool
        Whether to substitute environment variables in config values.
        Defaults to True.

    Returns
    -------
    Dict[str, Any]
        The loaded and validated configuration dictionary.

    Raises
    ------
    FileNotFoundError
        If the configuration file does not exist.
    json.JSONDecodeError
        If the file contains invalid JSON.
    ValueError
        If graph_definition validation fails or env var substitution fails.
    """
    # Resolve path
    if base_dir and not os.path.isabs(config_path):
        resolved_path = os.path.join(base_dir, config_path)
    else:
        resolved_path = config_path

    # Load JSON
    with open(resolved_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    # Substitute environment variables
    if substitute_env:
        config = substitute_env_vars(config)

    # Validate graph definition if present
    if "graph_definition" in config:
        config["graph_definition"] = validate_graph_definition(config["graph_definition"])

    return config


def load_graph_definition(
    config: Dict[str, Any],
    validate: bool = True,
) -> Dict[str, Any]:
    """Extract and optionally validate the graph definition from a config.

    Parameters
    ----------
    config : Dict[str, Any]
        The full application configuration.
    validate : bool
        Whether to validate the graph definition. Defaults to True.

    Returns
    -------
    Dict[str, Any]
        The graph definition dictionary containing 'nodes' and 'edges'.

    Raises
    ------
    KeyError
        If 'graph_definition' key is not present in config.
    ValueError
        If validation fails.
    """
    if "graph_definition" not in config:
        raise KeyError("Configuration must contain 'graph_definition' key.")

    graph_def = config["graph_definition"]

    if validate:
        graph_def = validate_graph_definition(graph_def)

    return graph_def


def get_system_variables(config: Dict[str, Any]) -> Dict[str, Any]:
    """Extract system variables from configuration.

    Parameters
    ----------
    config : Dict[str, Any]
        The full application configuration.

    Returns
    -------
    Dict[str, Any]
        The system_variables dictionary, or empty dict if not present.
    """
    return config.get("system_variables", {})


def get_agent_prompts(config: Dict[str, Any]) -> Dict[str, str]:
    """Extract agent prompts from configuration.

    Parameters
    ----------
    config : Dict[str, Any]
        The full application configuration.

    Returns
    -------
    Dict[str, str]
        The agent_prompts dictionary, or empty dict if not present.
    """
    return config.get("agent_prompts", {})
