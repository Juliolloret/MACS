from __future__ import annotations

from typing import Any, Dict


def mutate_graph_definition(graph_definition: Dict[str, Any], step: int) -> Dict[str, Any]:
    """Return a mutated copy of ``graph_definition`` for the given ``step``.

    This placeholder implementation simply annotates the graph with the current
    evolution step. Real mutation logic can be inserted here.
    """
    mutated = dict(graph_definition)
    metadata = dict(mutated.get("metadata", {}))
    metadata["evolution_step"] = step
    mutated["metadata"] = metadata
    return mutated
