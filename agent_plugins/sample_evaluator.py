"""Example evaluation plugin for tests and documentation."""

from __future__ import annotations

from typing import Any, Dict

from adaptive.evaluation import EvaluationPlugin


class SampleEvaluator(EvaluationPlugin):
    """Simple evaluator that reads a ``score`` value from outputs."""

    def evaluate(
        self,
        outputs: Dict[str, Any],
        graph_def: Dict[str, Any],
        threshold: float,
        step: int,
    ) -> float:
        return float(outputs.get("score", 0.0))
