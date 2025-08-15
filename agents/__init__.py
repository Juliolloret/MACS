from .registry import AGENT_REGISTRY, register_agent, get_agent_class, load_agents
from .base_agent import Agent
from .sdk_models import WebSearchItem, WebSearchPlan, ReportData

# Import all modules ending with '_agent' to populate AGENT_REGISTRY
load_agents(__name__)

# Expose agent classes for backward compatibility
globals().update(AGENT_REGISTRY)

__all__ = ["Agent", "register_agent", "get_agent_class", "WebSearchItem", "WebSearchPlan", "ReportData"] + list(AGENT_REGISTRY.keys())
