from __future__ import annotations

"""Re-run a graph after evaluating experimental results.

This module provides a small utility for continuing an adaptive experimentation
cycle once laboratory results have been gathered.  The evaluation results are
recorded in a simple JSON "database" and the graph is executed again to propose
new experimental conditions.
"""

import argparse
from typing import Any, Dict

from .adaptive_graph_runner import adaptive_cycle
from .evaluation import evaluate_target_function
from .json_utils import load_json, save_json


def update_database(path: str, entry: Dict[str, Any]) -> None:
    """Append ``entry`` to the JSON database at ``path``.

    If the file does not exist, it is created with a top level ``experiments``
    list.  Each ``entry`` typically contains the experimental conditions and
    the measured property (e.g. yield or selectivity).
    """
    try:
        db = load_json(path)
    except FileNotFoundError:
        db = {"experiments": []}
    db.setdefault("experiments", []).append(entry)
    save_json(path, db)


def rerun_with_evaluation(
    config_path: str,
    inputs_path: str,
    evaluation_path: str,
    database_path: str,
    *,
    threshold: float,
    max_steps: int,
) -> None:
    """Update the experiment database and run the adaptive cycle again."""
    run_inputs = load_json(inputs_path)
    evaluation_result = load_json(evaluation_path)
    update_database(database_path, evaluation_result)

    adaptive_cycle(
        config_path,
        run_inputs,
        evaluate_target_function,
        threshold=threshold,
        max_steps=max_steps,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Re-run the graph after evaluating experiments."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.json",
        help="Path to configuration file.",
    )
    parser.add_argument(
        "--inputs",
        type=str,
        required=True,
        help="Path to JSON file with 'initial_inputs' and 'project_base_output_dir'.",
    )
    parser.add_argument(
        "--evaluation",
        type=str,
        required=True,
        help="Path to JSON file containing the evaluation results.",
    )
    parser.add_argument(
        "--database",
        type=str,
        required=True,
        help="Path to the experiments database JSON file.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=1.0,
        help="Evaluation threshold for termination.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=5,
        help="Maximum number of adaptation steps.",
    )
    args = parser.parse_args()

    rerun_with_evaluation(
        args.config,
        args.inputs,
        args.evaluation,
        args.database,
        threshold=args.threshold,
        max_steps=args.max_steps,
    )
