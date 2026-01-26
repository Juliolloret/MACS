"""Tests for the EventBus and orchestrator event emission."""

import os
import shutil
import sys
import unittest
from unittest.mock import patch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from events import (
    EventBus,
    EVENT_NODE_COMPLETED,
    EVENT_NODE_STARTED,
    EVENT_WORKFLOW_COMPLETED,
    EVENT_WORKFLOW_STARTED,
)
from llm_fake import FakeLLM
from agents.base_agent import Agent
from agents.registry import register_agent
from multi_agent_llm_system import GraphOrchestrator


@register_agent("EventBusAgentA")
class EventBusAgentA(Agent):
    """Simple agent returning a fixed output."""

    def execute(self, _inputs):
        return {"out_a": "ok"}


@register_agent("EventBusAgentB")
class EventBusAgentB(Agent):
    """Simple agent returning a fixed output."""

    def execute(self, _inputs):
        return {"out_b": "ok"}


class TestEventBus(unittest.TestCase):
    """Tests covering the event bus behavior and integration."""

    def setUp(self):
        self.test_outputs_dir = "test_outputs_event_bus"
        os.makedirs(self.test_outputs_dir, exist_ok=True)

    def tearDown(self):
        if os.path.exists(self.test_outputs_dir):
            shutil.rmtree(self.test_outputs_dir)

    def test_event_bus_specific_and_wildcard_handlers(self):
        """EventBus dispatches to specific and wildcard subscribers."""
        bus = EventBus()
        seen = []
        wildcard_seen = []

        bus.subscribe("alpha", seen.append)
        bus.subscribe("*", wildcard_seen.append)

        bus.emit("alpha", {"value": 1})

        self.assertEqual(len(seen), 1)
        self.assertEqual(seen[0].name, "alpha")
        self.assertEqual(seen[0].payload.get("value"), 1)
        self.assertEqual(len(wildcard_seen), 1)
        self.assertEqual(wildcard_seen[0].name, "alpha")

    def test_orchestrator_emits_node_events(self):
        """GraphOrchestrator emits lifecycle events via the bus."""
        bus = EventBus()
        captured = []
        bus.subscribe("*", captured.append)

        config = {
            "graph_definition": {
                "nodes": [
                    {"id": "a", "type": "EventBusAgentA"},
                    {"id": "b", "type": "EventBusAgentB"},
                ],
                "edges": [
                    {"from": "a", "to": "b", "data_mapping": {"out_a": "in_a"}},
                ],
            }
        }
        llm = FakeLLM()
        app_config = {"system_variables": {"default_llm_model": "test_model"}}
        orchestrator = GraphOrchestrator(
            config["graph_definition"],
            llm,
            app_config,
            event_bus=bus,
        )

        with patch.object(GraphOrchestrator, "visualize", return_value=""):
            orchestrator.run({}, self.test_outputs_dir)

        names = [event.name for event in captured]
        self.assertIn(EVENT_WORKFLOW_STARTED, names)
        self.assertIn(EVENT_WORKFLOW_COMPLETED, names)

        started = [event.payload.get("node_id") for event in captured if event.name == EVENT_NODE_STARTED]
        completed = [event.payload.get("node_id") for event in captured if event.name == EVENT_NODE_COMPLETED]
        self.assertEqual(started, ["a", "b"])
        self.assertEqual(completed, ["a", "b"])

