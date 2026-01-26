"""Tests for execution_engine module."""

import pytest
from unittest import mock

from execution_engine import NodeExecutor


class MockAgent:
    """Mock agent for testing."""

    def __init__(self, responses=None, should_fail=False):
        self.responses = responses or [{"output": "success"}]
        self.should_fail = should_fail
        self.call_count = 0
        self.last_inputs = None

    def execute(self, inputs):
        self.last_inputs = inputs
        self.call_count += 1

        if self.should_fail:
            raise RuntimeError("Simulated failure")

        if self.call_count <= len(self.responses):
            return self.responses[self.call_count - 1]
        return self.responses[-1]


class TestNodeExecutor:
    """Tests for NodeExecutor class."""

    def test_execute_simple_node(self):
        """Test executing a simple node."""
        agent = MockAgent([{"result": "done"}])
        executor = NodeExecutor()

        result = executor.execute_node(
            agent, "test_node", {"input": "data"}, {}
        )

        assert result == {"result": "done"}
        assert agent.call_count == 1

    def test_execute_with_retry_on_failure(self):
        """Test that retries work when configured."""
        # First call fails, second succeeds
        agent = MockAgent([{"error": "failed"}, {"result": "success"}])
        retry_calls = []

        def on_retry(node_id, attempt, error):
            retry_calls.append((node_id, attempt, error))

        executor = NodeExecutor(on_retry=on_retry)

        result = executor.execute_node(
            agent,
            "test_node",
            {"input": "data"},
            {"failure_policy": "retry", "retries": 1}
        )

        assert result == {"result": "success"}
        assert agent.call_count == 2
        assert len(retry_calls) == 1
        assert retry_calls[0][0] == "test_node"

    def test_execute_no_retry_on_continue_policy(self):
        """Test that continue policy doesn't retry."""
        agent = MockAgent([{"error": "failed"}])
        executor = NodeExecutor()

        result = executor.execute_node(
            agent,
            "test_node",
            {"input": "data"},
            {"failure_policy": "continue"}
        )

        assert result == {"error": "failed"}
        assert agent.call_count == 1

    def test_execute_catches_exceptions(self):
        """Test that exceptions are caught and returned as errors."""
        agent = MockAgent(should_fail=True)
        executor = NodeExecutor()

        result = executor.execute_node(agent, "test_node", {}, {})

        assert "error" in result
        assert "Simulated failure" in result["error"]


class TestLoopExecution:
    """Tests for loop execution functionality."""

    def test_loop_over_list(self):
        """Test executing agent in a loop over items."""
        agent = MockAgent([
            {"output": "result_1"},
            {"output": "result_2"},
            {"output": "result_3"},
        ])
        executor = NodeExecutor()

        result = executor.execute_node(
            agent,
            "test_node",
            {"items": ["a", "b", "c"]},
            {"loop_over": "items", "loop_item_input_key": "current_item"}
        )

        assert "results" in result
        assert len(result["results"]) == 3
        assert agent.call_count == 3

    def test_loop_fallback_to_initial_inputs(self):
        """Test that loop_over falls back to initial_inputs."""
        agent = MockAgent([{"output": "result"}])
        executor = NodeExecutor()

        result = executor.execute_node(
            agent,
            "test_node",
            {},  # loop_over key not in inputs
            {"loop_over": "items"},
            initial_inputs={"items": ["a"]}  # but is in initial_inputs
        )

        assert "results" in result
        assert agent.call_count == 1

    def test_parallel_execution(self):
        """Test parallel loop execution."""
        results = []

        class ParallelAgent:
            def execute(self, inputs):
                results.append(inputs.get("item"))
                return {"processed": inputs.get("item")}

        agent = ParallelAgent()
        executor = NodeExecutor()

        result = executor.execute_node(
            agent,
            "test_node",
            {"items": [1, 2, 3]},
            {"loop_over": "items", "parallel_execution": True, "max_workers": 2}
        )

        assert "results" in result
        assert len(result["results"]) == 3
        # All items should have been processed
        assert set(results) == {1, 2, 3}

    def test_loop_with_dict_items(self):
        """Test loop where items are dicts and should be merged."""
        agent = MockAgent([{"output": "done"}])
        executor = NodeExecutor()

        result = executor.execute_node(
            agent,
            "test_node",
            {"items": [{"key": "val1"}]},
            {"loop_over": "items"}  # no loop_item_input_key, so dict merges
        )

        assert agent.last_inputs.get("key") == "val1"


class TestBuildIterationInputs:
    """Tests for _build_iteration_inputs."""

    def test_with_item_input_key(self):
        """Test building inputs with explicit item key."""
        result = NodeExecutor._build_iteration_inputs(
            {"base": "data"}, "item_value", "my_item", "loop_key"
        )

        assert result == {"base": "data", "my_item": "item_value"}

    def test_with_dict_item(self):
        """Test building inputs with dict item (merge)."""
        result = NodeExecutor._build_iteration_inputs(
            {"base": "data"}, {"extra": "value"}, None, "loop_key"
        )

        assert result == {"base": "data", "extra": "value"}

    def test_with_scalar_item(self):
        """Test building inputs with scalar item."""
        result = NodeExecutor._build_iteration_inputs(
            {"base": "data"}, 42, None, "loop_key"
        )

        assert result == {"base": "data", "item": 42}

    def test_excludes_loop_key(self):
        """Test that loop_over key is excluded."""
        result = NodeExecutor._build_iteration_inputs(
            {"base": "data", "items": [1, 2, 3]}, "x", "item", "items"
        )

        assert "items" not in result
