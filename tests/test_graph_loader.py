"""Tests for graph_loader module."""

import json
import os
import pytest
from unittest import mock

from graph_loader import (
    substitute_env_vars,
    load_config,
    load_graph_definition,
    get_system_variables,
    get_agent_prompts,
)


class TestSubstituteEnvVars:
    """Tests for environment variable substitution."""

    def test_substitute_simple_string(self):
        """Test substituting a simple environment variable."""
        with mock.patch.dict(os.environ, {"MY_VAR": "test_value"}):
            result = substitute_env_vars("prefix_${MY_VAR}_suffix")
            assert result == "prefix_test_value_suffix"

    def test_substitute_with_default(self):
        """Test substitution with default value when var is not set."""
        # Ensure the variable is not set
        with mock.patch.dict(os.environ, {}, clear=True):
            os.environ.pop("UNSET_VAR", None)
            result = substitute_env_vars("${UNSET_VAR:-default_value}")
            assert result == "default_value"

    def test_substitute_with_default_when_set(self):
        """Test that value is used when variable is set."""
        with mock.patch.dict(os.environ, {"SET_VAR": "actual_value"}):
            result = substitute_env_vars("${SET_VAR:-default_value}")
            assert result == "actual_value"

    def test_substitute_missing_var_raises(self):
        """Test that missing variable without default raises ValueError."""
        with mock.patch.dict(os.environ, {}, clear=True):
            os.environ.pop("MISSING_VAR", None)
            with pytest.raises(ValueError, match="MISSING_VAR"):
                substitute_env_vars("${MISSING_VAR}")

    def test_substitute_in_dict(self):
        """Test recursive substitution in dictionary."""
        with mock.patch.dict(os.environ, {"API_KEY": "secret123"}):
            config = {
                "key": "${API_KEY}",
                "nested": {
                    "value": "${API_KEY}"
                }
            }
            result = substitute_env_vars(config)
            assert result["key"] == "secret123"
            assert result["nested"]["value"] == "secret123"

    def test_substitute_in_list(self):
        """Test recursive substitution in list."""
        with mock.patch.dict(os.environ, {"ITEM": "value"}):
            items = ["${ITEM}", "plain", {"nested": "${ITEM}"}]
            result = substitute_env_vars(items)
            assert result == ["value", "plain", {"nested": "value"}]

    def test_no_substitution_for_non_string(self):
        """Test that non-string values pass through unchanged."""
        assert substitute_env_vars(123) == 123
        assert substitute_env_vars(True) is True
        assert substitute_env_vars(None) is None


class TestLoadConfig:
    """Tests for configuration loading."""

    def test_load_valid_config(self, tmp_path):
        """Test loading a valid configuration file."""
        config_data = {
            "system_variables": {
                "default_llm_model": "gpt-4"
            },
            "graph_definition": {
                "nodes": [{"id": "node1", "type": "TestAgent"}],
                "edges": []
            }
        }
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps(config_data))

        result = load_config(str(config_file), substitute_env=False)

        assert result["system_variables"]["default_llm_model"] == "gpt-4"
        assert "graph_definition" in result

    def test_load_config_with_base_dir(self, tmp_path):
        """Test loading config with relative path resolution."""
        config_data = {"key": "value"}
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps(config_data))

        result = load_config("config.json", base_dir=str(tmp_path), substitute_env=False)

        assert result["key"] == "value"

    def test_load_config_with_env_substitution(self, tmp_path):
        """Test that env vars are substituted during load."""
        config_data = {"api_key": "${TEST_API_KEY}"}
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps(config_data))

        with mock.patch.dict(os.environ, {"TEST_API_KEY": "my_secret"}):
            result = load_config(str(config_file), substitute_env=True)

        assert result["api_key"] == "my_secret"

    def test_load_config_file_not_found(self):
        """Test that FileNotFoundError is raised for missing file."""
        with pytest.raises(FileNotFoundError):
            load_config("/nonexistent/path/config.json")

    def test_load_config_invalid_json(self, tmp_path):
        """Test that JSONDecodeError is raised for invalid JSON."""
        config_file = tmp_path / "config.json"
        config_file.write_text("not valid json {")

        with pytest.raises(json.JSONDecodeError):
            load_config(str(config_file))


class TestLoadGraphDefinition:
    """Tests for graph definition extraction."""

    def test_load_graph_definition(self):
        """Test extracting graph definition from config."""
        config = {
            "graph_definition": {
                "nodes": [{"id": "node1", "type": "TestAgent"}],
                "edges": []
            }
        }
        result = load_graph_definition(config, validate=True)

        assert "nodes" in result
        assert "edges" in result

    def test_load_graph_definition_missing_key(self):
        """Test that KeyError is raised when graph_definition is missing."""
        config = {"other_key": "value"}

        with pytest.raises(KeyError, match="graph_definition"):
            load_graph_definition(config)


class TestHelperFunctions:
    """Tests for helper functions."""

    def test_get_system_variables(self):
        """Test extracting system variables."""
        config = {
            "system_variables": {"key": "value"},
            "other": "data"
        }
        result = get_system_variables(config)
        assert result == {"key": "value"}

    def test_get_system_variables_missing(self):
        """Test that empty dict is returned when system_variables is missing."""
        config = {"other": "data"}
        result = get_system_variables(config)
        assert result == {}

    def test_get_agent_prompts(self):
        """Test extracting agent prompts."""
        config = {
            "agent_prompts": {"prompt1": "text"}
        }
        result = get_agent_prompts(config)
        assert result == {"prompt1": "text"}

    def test_get_agent_prompts_missing(self):
        """Test that empty dict is returned when agent_prompts is missing."""
        config = {}
        result = get_agent_prompts(config)
        assert result == {}
