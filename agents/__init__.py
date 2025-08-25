from .registry import AGENT_REGISTRY, register_agent, get_agent_class, load_agents
from .base_agent import Agent
from .memory_agent import ShortTermMemoryAgent, LongTermMemoryAgent
from .deep_research_summarizer_agent import DeepResearchSummarizerAgent

try:  # Optional models used only by SDK-based agents
    from .sdk_models import WebSearchItem, WebSearchPlan, ReportData
    _sdk_models = ["WebSearchItem", "WebSearchPlan", "ReportData"]
except Exception:  # pragma: no cover
    _sdk_models = []

# Import all modules ending with '_agent' to populate AGENT_REGISTRY
load_agents(__name__)

# -- Agent classes are now loaded into the registry via load_agents() and accessed via get_agent_class() --
# -- The explicit import and exposure of all agent classes is removed to simplify the package interface --

__all__ = [
    "Agent",
    "AGENT_REGISTRY",
    "register_agent",
    "get_agent_class",
    "load_agents",
    "ShortTermMemoryAgent",
    "LongTermMemoryAgent",
    "DeepResearchSummarizerAgent",
] + _sdk_models
