"""Utilities for running an adaptive agent graph.

This module provides a command line interface and helper functions for executing
and iteratively adapting an agent graph based on evaluation feedback. The
``adaptive_cycle`` function orchestrates the graph execution and optionally
updates the graph definition after each iteration.
"""

from __future__ import annotations

import argparse
import os
from typing import Any, Dict, List

from llm_fake import FakeLLM
from llm_openai import OpenAILLM
from multi_agent_llm_system import GraphOrchestrator

from .evaluation import load_evaluation_plugins
from .mutation import mutate_graph_definition
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
    *,
    threshold: float,
    max_steps: int,
) -> Dict[str, Any]:
    """Run the adaptive orchestration cycle."""
    config = load_json(config_path)
    graph_definition = config.get("graph_definition", {})
    evaluators = load_evaluation_plugins(config.get("evaluation_plugins"))
    final_outputs: Dict[str, Any] = {}

    history_dir = os.path.join(os.path.dirname(config_path), "config_history")
    os.makedirs(history_dir, exist_ok=True)
    save_json(os.path.join(history_dir, "config_1.json"), config)

    cycle_records: List[Dict[str, Any]] = []

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

        metric_scores: Dict[str, float] = {}
        for evaluator in evaluators:
            score = evaluator.evaluate(final_outputs, graph_definition, threshold, step)
            metric_scores[evaluator.__class__.__name__] = score
        if not metric_scores:
            metric_scores["score"] = float(final_outputs.get("score", 0.0))
        scores = list(metric_scores.values())
        aggregated = sum(scores) / len(scores)

        cycle_records.append(
            {
                "cycle_id": step,
                "config_file": f"config_{step}.json",
                "results": final_outputs,
                "metrics": metric_scores,
                "aggregated_score": aggregated,
            }
        )

        if aggregated >= threshold:
            break

        graph_definition = mutate_graph_definition(graph_definition, step)
        config["graph_definition"] = graph_definition
        save_json(os.path.join(history_dir, f"config_{step + 1}.json"), config)

    ranked = sorted(cycle_records, key=lambda r: r["aggregated_score"], reverse=True)
    for idx, rec in enumerate(ranked, 1):
        rec["rank"] = idx
    save_json(os.path.join(history_dir, "cycles.json"), ranked)

    return final_outputs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Adaptive Graph Runner")
    parser.add_argument("--config", type=str, default="config.json", help="Path to configuration file.")
    parser.add_argument(
        "--inputs",
        type=str,
        required=True,
        help="Path to JSON file with 'initial_inputs' and 'project_base_output_dir'.",
    )
    parser.add_argument("--threshold", type=float, default=1.0, help="Evaluation threshold for termination.")
    parser.add_argument("--max-steps", type=int, default=5, help="Maximum number of adaptation steps.")
    args = parser.parse_args()

    run_inputs = load_json(args.inputs)
    adaptive_cycle(
        args.config,
        run_inputs,
        threshold=args.threshold,
        max_steps=args.max_steps,
    )
