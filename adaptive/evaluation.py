"""Helper functions and plugin interfaces for adaptive graph evaluation.

This module exposes :class:`EvaluationPlugin` used for pluggable evaluation
strategies and :func:`evaluate_target_function`, a small utility used by the
adaptive workflow to decide whether further mutation of the graph is required
based on the score returned from a run.
"""

from __future__ import annotations

import importlib
from importlib import metadata
from typing import Any, Dict, List, Optional, Protocol, Sequence

from .mutation import mutate_graph_definition


class EvaluationPlugin(Protocol):
    """Protocol for evaluation plugins.

    Evaluation plugins analyse the outputs of a graph run and return a numeric
    score. The adaptive runner aggregates scores from all loaded plugins to
    decide whether further mutation is required.
    """

    def evaluate(
        self,
        outputs: Dict[str, Any],
        graph_def: Dict[str, Any],
        threshold: float,
        step: int,
    ) -> float:
        """Return a score for the current step."""


def load_evaluation_plugins(paths: Optional[Sequence[str]] = None) -> List[EvaluationPlugin]:
    """Load evaluation plugins from module paths or entry points.

    Parameters
    ----------
    paths:
        Optional sequence of ``"module:Class"`` strings pointing to evaluator
        classes defined in configuration files.
    """

    evaluators: List[EvaluationPlugin] = []

    for path in paths or []:
        module_name, class_name = path.split(":")
        module = importlib.import_module(module_name)
        cls = getattr(module, class_name)
        evaluators.append(cls())

    try:
        entry_points = metadata.entry_points()
        if hasattr(entry_points, "select"):
            eval_eps = entry_points.select(group="macs.evaluators")
        else:  # pragma: no cover - deprecated API
            eval_eps = entry_points.get("macs.evaluators", [])
        for ep in eval_eps:
            try:
                cls = ep.load()
                evaluators.append(cls())
            except Exception:  # pragma: no cover - defensive
                continue
    except Exception:  # pragma: no cover - defensive
        pass

    return evaluators


def evaluate_target_function(
    outputs: Dict[str, Any],
    graph_definition: Dict[str, Any],
    threshold: float,
    step: int,
) -> Optional[Dict[str, Any]]:
    """Evaluate ``outputs`` and decide whether to mutate the graph.

    Parameters
    ----------
    outputs:
        The result from the graph execution.
    graph_definition:
        Current graph definition.
    threshold:
        Desired score threshold to terminate the adaptation.
    step:
        Current evolution step.

    Returns
    -------
    Optional[Dict[str, Any]]
        A new graph definition if further adaptation is required or ``None`` to
        stop the cycle.
    """
    score = outputs.get("score")
    if score is not None and score >= threshold:
        return None
    return mutate_graph_definition(graph_definition, step)
