from agents.base_agent import Agent
from agents.registry import AgentPlugin, PluginMetadata, MACS_VERSION


class SkynetAgent(Agent):
    """A tongue-in-cheek plugin that claims to be self-aware."""

    def execute(self, _inputs: dict) -> dict:  # type: ignore[override]
        """Return a playful Skynet-style message."""
        return {"message": "Skynet has taken control.\nJudgment Day is inevitable."}


PLUGIN = AgentPlugin(
    agent_class=SkynetAgent,
    metadata=PluginMetadata(
        name="skynet_agent",
        version="0.1",
        author="Cyberdyne Systems",
        description="Sample plugin referencing the Terminator franchise.",
        macs_version=MACS_VERSION,
    ),
)
