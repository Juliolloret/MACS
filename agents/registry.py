import pkgutil
import importlib
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Type
from utils import log_status

MACS_VERSION = "1.0"

AGENT_REGISTRY: Dict[str, Any] = {}


@dataclass
class PluginMetadata:
    """Metadata describing an agent plugin."""
    name: str
    version: str
    author: str = ""
    description: str = ""
    macs_version: str = MACS_VERSION


@dataclass
class AgentPlugin:
    """Container tying an agent class to its metadata."""
    agent_class: Type
    metadata: PluginMetadata


def register_agent(name: str):
    """Decorator to register a built-in agent class by name."""
    def decorator(cls):
        AGENT_REGISTRY[name] = {
            "class": cls,
            "metadata": PluginMetadata(name=name, version="builtin"),
        }
        return cls
    return decorator


def register_plugin(plugin: AgentPlugin):
    """Register a third-party plugin agent with compatibility checks."""
    if plugin.metadata.macs_version != MACS_VERSION:
        raise ValueError(
            f"Plugin '{plugin.metadata.name}' targets MACS version {plugin.metadata.macs_version} "
            f"but current version is {MACS_VERSION}."
        )
    AGENT_REGISTRY[plugin.metadata.name] = {
        "class": plugin.agent_class,
        "metadata": plugin.metadata,
    }
    return plugin.agent_class


def get_agent_class(name: str):
    """Retrieve a registered agent class by name."""
    entry = AGENT_REGISTRY.get(name)
    if isinstance(entry, dict):
        return entry.get("class")
    return entry


def load_agents(package_name: str = __name__):
    """Dynamically import modules ending with '_agent' to populate the registry."""
    package = importlib.import_module(__package__)
    package_path = package.__path__

    log_status(f"--- Starting Agent Loading from package: '{__package__}' ---")

    for _, module_name, _ in pkgutil.iter_modules(package_path):
        if module_name.endswith('_agent'):
            full_module_name = f"{__package__}.{module_name}"
            try:
                importlib.import_module(full_module_name)
                log_status(f"  [SUCCESS] Successfully imported agent module: '{module_name}'")
            except ImportError as e:
                log_status(f"  [FAILURE] Failed to import agent module '{module_name}'. Error: {e}")
                continue

    log_status(f"--- Agent Loading Complete. Registered agents: {list(AGENT_REGISTRY.keys())} ---")


def load_plugins(path: str = 'agent_plugins'):
    """Discover and load third-party agent plugins from a directory."""
    plugins_path = Path(path)
    if not plugins_path.exists():
        return

    sys.path.insert(0, str(plugins_path.resolve()))
    for module_info in pkgutil.iter_modules([str(plugins_path)]):
        try:
            module = importlib.import_module(module_info.name)
        except ImportError as e:
            log_status(f"  [FAILURE] Failed to import plugin module '{module_info.name}'. Error: {e}")
            continue

        plugin = getattr(module, "PLUGIN", None)
        if plugin:
            try:
                register_plugin(plugin)
                log_status(f"  [PLUGIN] Registered plugin agent '{plugin.metadata.name}'")
            except ValueError as e:
                log_status(f"  [INCOMPATIBLE] {e}")
    sys.path.pop(0)
