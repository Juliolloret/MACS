"""Node execution utilities for loop handling, retries, and parallel execution."""

from __future__ import annotations

import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Dict, List, Optional

from utils import log_status


class NodeExecutor:
    """Execute agent nodes with retry, loop, and parallel support.

    This class extracts the node execution logic from the orchestrator
    to make it easier to test and extend.
    """

    def __init__(
        self,
        default_failure_policy: str = "continue",
        on_retry: Optional[Callable[[str, int, str], None]] = None,
    ):
        """Initialize the executor.

        Parameters
        ----------
        default_failure_policy : str
            Default policy when nodes fail. One of 'continue', 'abort', 'retry'.
        on_retry : Optional[Callable]
            Callback invoked when a retry occurs. Receives (node_id, attempt, error).
        """
        self.default_failure_policy = default_failure_policy
        self.on_retry = on_retry

    def execute_node(
        self,
        agent: Any,
        node_id: str,
        inputs: Dict[str, Any],
        config: Dict[str, Any],
        initial_inputs: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Execute a single node with retry and loop support.

        Parameters
        ----------
        agent : Any
            The agent instance to execute.
        node_id : str
            Identifier of the node being executed.
        inputs : Dict[str, Any]
            Input data gathered for this node.
        config : Dict[str, Any]
            Node configuration (from config_params).
        initial_inputs : Optional[Dict[str, Any]]
            Initial workflow inputs (for loop_over fallback).

        Returns
        -------
        Dict[str, Any]
            Node output dictionary.
        """
        policy = config.get("failure_policy", self.default_failure_policy)
        retries = config.get("retries", 0)
        max_attempts = retries + 1 if policy == "retry" else 1

        node_output = {}
        for attempt in range(1, max_attempts + 1):
            try:
                node_output = self._execute_with_loop(
                    agent, node_id, inputs, config, initial_inputs
                )
            except Exception as exc:  # pylint: disable=broad-exception-caught
                detailed_traceback = traceback.format_exc()
                log_status(
                    f"[NodeExecutor] CRITICAL_ERROR: Node '{node_id}' failed: "
                    f"{exc}\n{detailed_traceback}"
                )
                node_output = {"error": f"Agent execution for '{node_id}' failed: {exc}"}

            # Check if we should retry
            if node_output.get("error") and policy == "retry" and attempt < max_attempts:
                log_status(f"[{node_id}] RETRY_ATTEMPT_{attempt} failed; retrying...")
                if self.on_retry:
                    self.on_retry(node_id, attempt, node_output.get("error", ""))
                continue
            break

        return node_output

    def _execute_with_loop(
        self,
        agent: Any,
        node_id: str,
        inputs: Dict[str, Any],
        config: Dict[str, Any],
        initial_inputs: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Execute agent, handling loop_over if configured.

        Parameters
        ----------
        agent : Any
            The agent instance to execute.
        node_id : str
            Identifier of the node being executed.
        inputs : Dict[str, Any]
            Input data gathered for this node.
        config : Dict[str, Any]
            Node configuration.
        initial_inputs : Optional[Dict[str, Any]]
            Initial workflow inputs.

        Returns
        -------
        Dict[str, Any]
            Node output dictionary.
        """
        loop_over_key = config.get("loop_over")
        initial_inputs = initial_inputs or {}

        # Fallback: get loop_over value from initial_inputs if not in inputs
        if loop_over_key and loop_over_key in initial_inputs and loop_over_key not in inputs:
            inputs[loop_over_key] = initial_inputs[loop_over_key]

        # Check if we should loop
        if (
            loop_over_key
            and loop_over_key in inputs
            and isinstance(inputs[loop_over_key], list)
        ):
            return self._execute_loop(agent, node_id, inputs, config, loop_over_key)

        # Standard single execution
        return agent.execute(inputs)

    def _execute_loop(
        self,
        agent: Any,
        node_id: str,
        inputs: Dict[str, Any],
        config: Dict[str, Any],
        loop_over_key: str,
    ) -> Dict[str, Any]:
        """Execute agent in a loop over input items.

        Parameters
        ----------
        agent : Any
            The agent instance to execute.
        node_id : str
            Identifier of the node being executed.
        inputs : Dict[str, Any]
            Input data including the loop items.
        config : Dict[str, Any]
            Node configuration.
        loop_over_key : str
            Key in inputs containing the list to iterate over.

        Returns
        -------
        Dict[str, Any]
            Dictionary with 'results' containing list of outputs.
        """
        input_list = inputs[loop_over_key]
        loop_outputs: List[Optional[Dict[str, Any]]] = [None] * len(input_list)
        item_input_key = config.get("loop_item_input_key")

        log_status(
            f"[{node_id}] LOOP_START: Iterating over {len(input_list)} items "
            f"from '{loop_over_key}'."
        )

        if config.get("parallel_execution"):
            loop_outputs = self._execute_parallel(
                agent, inputs, input_list, item_input_key, loop_over_key, config
            )
        else:
            loop_outputs = self._execute_sequential(
                agent, node_id, inputs, input_list, item_input_key, loop_over_key
            )

        return {"results": loop_outputs}

    def _execute_parallel(
        self,
        agent: Any,
        inputs: Dict[str, Any],
        input_list: List[Any],
        item_input_key: Optional[str],
        loop_over_key: str,
        config: Dict[str, Any],
    ) -> List[Optional[Dict[str, Any]]]:
        """Execute loop iterations in parallel.

        Parameters
        ----------
        agent : Any
            The agent instance to execute.
        inputs : Dict[str, Any]
            Base input data.
        input_list : List[Any]
            Items to iterate over.
        item_input_key : Optional[str]
            Key to use for each item in iteration inputs.
        loop_over_key : str
            Key to exclude from iteration inputs.
        config : Dict[str, Any]
            Node configuration.

        Returns
        -------
        List[Optional[Dict[str, Any]]]
            List of outputs from each iteration.
        """
        outputs: List[Optional[Dict[str, Any]]] = [None] * len(input_list)
        max_workers = config.get("max_workers")

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    agent.execute,
                    self._build_iteration_inputs(inputs, item, item_input_key, loop_over_key)
                ): idx
                for idx, item in enumerate(input_list)
            }

            for future in as_completed(futures):
                idx = futures[future]
                try:
                    outputs[idx] = future.result()
                except Exception as exc:  # pylint: disable=broad-exception-caught
                    outputs[idx] = {"error": f"Parallel execution failed: {exc}"}

        return outputs

    def _execute_sequential(
        self,
        agent: Any,
        node_id: str,
        inputs: Dict[str, Any],
        input_list: List[Any],
        item_input_key: Optional[str],
        loop_over_key: str,
    ) -> List[Optional[Dict[str, Any]]]:
        """Execute loop iterations sequentially.

        Parameters
        ----------
        agent : Any
            The agent instance to execute.
        node_id : str
            Identifier of the node.
        inputs : Dict[str, Any]
            Base input data.
        input_list : List[Any]
            Items to iterate over.
        item_input_key : Optional[str]
            Key to use for each item in iteration inputs.
        loop_over_key : str
            Key to exclude from iteration inputs.

        Returns
        -------
        List[Optional[Dict[str, Any]]]
            List of outputs from each iteration.
        """
        outputs: List[Optional[Dict[str, Any]]] = []

        for i, item in enumerate(input_list):
            log_status(f"[{node_id}] -> Loop {i + 1}/{len(input_list)}")
            iteration_inputs = self._build_iteration_inputs(
                inputs, item, item_input_key, loop_over_key
            )
            result = agent.execute(iteration_inputs)
            outputs.append(result)

        return outputs

    @staticmethod
    def _build_iteration_inputs(
        base_inputs: Dict[str, Any],
        item: Any,
        item_input_key: Optional[str],
        loop_over_key: str,
    ) -> Dict[str, Any]:
        """Build inputs for a single loop iteration.

        Parameters
        ----------
        base_inputs : Dict[str, Any]
            Original inputs (without loop items).
        item : Any
            Current item from the loop.
        item_input_key : Optional[str]
            Key to assign the item to.
        loop_over_key : str
            Key to exclude from iteration inputs.

        Returns
        -------
        Dict[str, Any]
            Inputs for this iteration.
        """
        iteration_inputs = {k: v for k, v in base_inputs.items() if k != loop_over_key}

        if item_input_key:
            iteration_inputs[item_input_key] = item
        elif isinstance(item, dict):
            iteration_inputs.update(item)
        else:
            iteration_inputs["item"] = item

        return iteration_inputs
