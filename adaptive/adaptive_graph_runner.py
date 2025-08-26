"""Utilities for running an adaptive agent graph.

This module provides a command line interface and helper functions for executing
and iteratively adapting an agent graph based on evaluation feedback. The
``adaptive_cycle`` function orchestrates the graph execution and optionally
updates the graph definition after each iteration.
"""

from __future__ import annotations

import argparse
import importlib
from typing import Any, Callable, Dict, Optional

from llm_fake import FakeLLM
from llm_openai import OpenAILLM
from multi_agent_llm_system import GraphOrchestrator

from .evaluation import evaluate_target_function
from .json_utils import load_json, save_json


def _create_llm_client(app_config: Dict[str, Any]):
    """Initialize the LLM client based on configuration."""
    system_vars = app_config.get("system_variables", {})
    llm_client_type = system_vars.get("llm_client", "openai")
    api_key = system_vars.get("openai_api_key")
    timeout = float(system_vars.get("openai_api_timeout_seconds", 120))

    if llm_client_type == "openai" and api_key and "YOUR" not in api_key:
        return OpenAILLM(app_config=app_config, api_key=api_key, timeout=int(timeout))
    return FakeLLM(app_config=app_config)


def adaptive_cycle(
    config_path: str,
    inputs: Dict[str, Any],
    evaluate_fn: Callable[[Dict[str, Any], Dict[str, Any], float, int], Optional[Dict[str, Any]]],
    *,
    threshold: float,
    max_steps: int,
) -> Dict[str, Any]:
    """Run the adaptive orchestration cycle."""
    config = load_json(config_path)
    graph_definition = config.get("graph_definition", {})
    final_outputs: Dict[str, Any] = {}

    for step in range(1, max_steps + 1):
        llm = _create_llm_client(config)
        orchestrator = GraphOrchestrator(graph_definition, llm, config)
        try:
            final_outputs = orchestrator.run(
                initial_inputs=inputs.get("initial_inputs", {}),
                project_base_output_dir=inputs.get("project_base_output_dir", "."),
            )
        finally:
            if hasattr(llm, "close"):
                llm.close()

        new_graph = evaluate_fn(final_outputs, graph_definition, threshold, step)
        if not new_graph:
            break

        graph_definition = new_graph
        config["graph_definition"] = graph_definition
        save_json(config_path, config)

    return final_outputs


def _resolve_callable(path: str) -> Callable[[Dict[str, Any], Dict[str, Any], float, int], Optional[Dict[str, Any]]]:
    """Resolve a callable from a ``module:function`` string."""
    module_name, func_name = path.split(":")
    module = importlib.import_module(module_name)
    return getattr(module, func_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Adaptive Graph Runner")
    parser.add_argument("--config", type=str, default="config.json", help="Path to configuration file.")
    parser.add_argument(
        "--inputs",
        type=str,
        required=True,
        help="Path to JSON file with 'initial_inputs' and 'project_base_output_dir'.",
    )
    parser.add_argument("--eval-hook", type=str, default="", help="Evaluation hook specified as 'module:function'.")
    parser.add_argument("--threshold", type=float, default=1.0, help="Evaluation threshold for termination.")
    parser.add_argument("--max-steps", type=int, default=5, help="Maximum number of adaptation steps.")
    args = parser.parse_args()

    run_inputs = load_json(args.inputs)
    evaluate_fn = evaluate_target_function if not args.eval_hook else _resolve_callable(args.eval_hook)

    adaptive_cycle(
        args.config,
        run_inputs,
        evaluate_fn,
        threshold=args.threshold,
        max_steps=args.max_steps,
    )
