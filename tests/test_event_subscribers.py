"""Tests for event bus subscribers."""

import os
import sys
import unittest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from events import (
    Event,
    EVENT_NODE_COMPLETED,
    EVENT_NODE_STARTED,
    EVENT_WORKFLOW_COMPLETED,
    EVENT_WORKFLOW_STARTED,
)
from event_subscribers import EventLogger, MetricsCollector


class TestEventLogger(unittest.TestCase):
    """Validate the default event logger subscriber."""

    def test_event_logger_emits_message(self):
        """EventLogger calls the configured logger with event details."""
        messages = []
        logger = EventLogger(logger=messages.append)
        logger.handle(Event(EVENT_NODE_STARTED, {"node_id": "node-a"}))

        self.assertEqual(len(messages), 1)
        self.assertIn(EVENT_NODE_STARTED, messages[0])
        self.assertIn("node-a", messages[0])


class TestMetricsCollector(unittest.TestCase):
    """Validate metrics aggregation from events."""

    def test_metrics_collector_tracks_counts_and_nodes(self):
        collector = MetricsCollector()
        collector.handle(Event(EVENT_WORKFLOW_STARTED, {"node_order": ["a", "b"]}))
        collector.handle(
            Event(
                EVENT_NODE_COMPLETED,
                {"node_id": "a", "elapsed_sec": 1.5, "tokens_used": 10, "error": None},
            )
        )
        collector.handle(
            Event(
                EVENT_NODE_COMPLETED,
                {"node_id": "b", "elapsed_sec": 2.0, "tokens_used": 5, "error": "boom"},
            )
        )
        collector.handle(Event(EVENT_WORKFLOW_COMPLETED, {"node_count": 2}))

        data = collector.to_dict()
        self.assertEqual(data["event_counts"][EVENT_NODE_COMPLETED], 2)
        self.assertEqual(data["node_metrics"]["a"]["elapsed_sec"], 1.5)
        self.assertEqual(data["node_metrics"]["b"]["error"], "boom")
        self.assertEqual(data["workflow"]["completed"]["node_count"], 2)
