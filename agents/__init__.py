from .registry import (
    AGENT_REGISTRY,
    AgentPlugin,
    PluginMetadata,
    MACS_VERSION,
    register_agent,
    register_plugin,
    get_agent_class,
    load_agents,
    load_plugins,
)
from .base_agent import Agent
from .memory_agent import ShortTermMemoryAgent, LongTermMemoryAgent
from .deep_research_summarizer_agent import DeepResearchSummarizerAgent

# Import all modules ending with '_agent' to populate AGENT_REGISTRY
load_agents(__name__)
# Load third-party plugins if any
load_plugins()

# -- Agent classes are now loaded into the registry via load_agents() and accessed via get_agent_class() --
# -- The explicit import and exposure of all agent classes is removed to simplify the package interface --

__all__ = [
    "Agent",
    "AGENT_REGISTRY",
    "AgentPlugin",
    "PluginMetadata",
    "MACS_VERSION",
    "register_agent",
    "register_plugin",
    "get_agent_class",
    "load_agents",
    "load_plugins",
    "ShortTermMemoryAgent",
    "LongTermMemoryAgent",
    "DeepResearchSummarizerAgent",
]
