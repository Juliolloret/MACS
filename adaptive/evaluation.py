from __future__ import annotations

from typing import Any, Dict, Optional

from .mutation import mutate_graph_definition


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
