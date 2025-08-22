from .registry import AGENT_REGISTRY, register_agent, get_agent_class, load_agents
from .base_agent import Agent

try:  # Optional models used only by SDK-based agents
    from .sdk_models import WebSearchItem, WebSearchPlan, ReportData
    _sdk_models = ["WebSearchItem", "WebSearchPlan", "ReportData"]
except Exception:  # pragma: no cover
    _sdk_models = []

# Import all modules ending with '_agent' to populate AGENT_REGISTRY
load_agents(__name__)

# Expose agent classes for backward compatibility
globals().update(AGENT_REGISTRY)

__all__ = ["Agent", "register_agent", "get_agent_class"] + _sdk_models + list(AGENT_REGISTRY.keys())
