"""Tests for the plugin registration and loading utilities."""

import pytest  # pylint: disable=import-error
from agents.base_agent import Agent
import importlib.metadata as md
from agents.registry import (
    AGENT_REGISTRY,
    AgentPlugin,
    PluginMetadata,
    register_plugin,
    load_plugins,
    get_agent_class,
    list_plugins,
)


def test_register_and_load_plugin(tmp_path):
    """Plugins can be registered from a directory and used."""
    plugin_dir = tmp_path / "plug"
    plugin_dir.mkdir()
    (plugin_dir / "__init__.py").write_text("")
    plugin_code = """
from agents.base_agent import Agent
from agents.registry import AgentPlugin, PluginMetadata, MACS_VERSION

class MyPluginAgent(Agent):
    def __init__(self, *args, **kwargs):
        pass

    def run(self):
        return "ok"

PLUGIN = AgentPlugin(
    agent_class=MyPluginAgent,
    metadata=PluginMetadata(name='my_plugin', version='0.1', macs_version=MACS_VERSION),
)
"""
    (plugin_dir / "my_plugin.py").write_text(plugin_code)

    prev_registry = AGENT_REGISTRY.copy()
    try:
        load_plugins(str(plugin_dir))
        assert "my_plugin" in AGENT_REGISTRY
        entry = AGENT_REGISTRY["my_plugin"]
        assert entry["metadata"].version == "0.1"
        cls = get_agent_class("my_plugin")
        assert issubclass(cls, Agent)
    finally:
        AGENT_REGISTRY.clear()
        AGENT_REGISTRY.update(prev_registry)


def test_incompatible_plugin_raises():
    """An incompatible plugin version results in a ``ValueError``."""

    class DummyAgent(Agent):
        """Minimal stub for an incompatible plugin."""

        pass

    plugin = AgentPlugin(
        agent_class=DummyAgent,
        metadata=PluginMetadata(name="bad_plugin", version="0.1", macs_version="0.0"),
    )

    with pytest.raises(ValueError):
        register_plugin(plugin)


def test_list_plugins_reports_registered(monkeypatch):
    """`list_plugins` returns metadata for registered agents."""

    class DummyAgent(Agent):
        pass

    plugin = AgentPlugin(
        agent_class=DummyAgent,
        metadata=PluginMetadata(name="dummy", version="0.1", author="me"),
    )
    prev_registry = AGENT_REGISTRY.copy()
    try:
        register_plugin(plugin)
        info = list_plugins()
        assert {"name": "dummy", "version": "0.1", "author": "me"} in info
    finally:
        AGENT_REGISTRY.clear()
        AGENT_REGISTRY.update(prev_registry)


def test_load_plugins_from_entry_points(monkeypatch):
    """Plugins exposed via entry points are discovered."""

    class EPAgent(Agent):
        pass

    ep_plugin = AgentPlugin(
        agent_class=EPAgent,
        metadata=PluginMetadata(name="ep_agent", version="0.2"),
    )

    class DummyEP:
        name = "ep_agent"

        def load(self):
            return ep_plugin

    def fake_entry_points():
        return {"macs.plugins": [DummyEP()]}

    monkeypatch.setattr(md, "entry_points", fake_entry_points)
    prev_registry = AGENT_REGISTRY.copy()
    try:
        load_plugins("nonexistent")
        assert "ep_agent" in AGENT_REGISTRY
    finally:
        AGENT_REGISTRY.clear()
        AGENT_REGISTRY.update(prev_registry)
