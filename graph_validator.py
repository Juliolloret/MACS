"""Graph validation mechanisms including schema checks and cycle detection."""

from typing import Any, Dict, List, Set, Optional

from config_schema import validate_graph_definition


class GraphValidationError(ValueError):
    """Raised when the graph structure is invalid."""
    pass


def validate_graph_structure(graph_def: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate the graph definition against the schema and check for structural issues.

    Checks:
    1. Schema validation (types, required fields)
    2. Reference validation (edges point to existing nodes)
    3. Cycle detection (graph must be a DAG)

    Parameters
    ----------
    graph_def : Dict[str, Any]
        The raw graph definition dictionary.

    Returns
    -------
    Dict[str, Any]
        The validated and normalized graph definition.

    Raises
    ------
    GraphValidationError
        If validation fails.
    """
    # 1. & 2. Schema validation
    try:
        validated_graph = validate_graph_definition(graph_def)
    except ValueError as e:
        raise GraphValidationError(f"Schema validation failed: {str(e)}") from e

    # 3. Cycle detection
    _detect_cycles(validated_graph)

    return validated_graph


def validate_node_inputs(
    graph_def: Dict[str, Any],
    agent_capabilities: Optional[Dict[str, Any]] = None
) -> List[str]:
    """
    Validate that nodes receive their required inputs.
    
    This is a preliminary check. Full validation requires instantiating agents
    to know their exact requirements, but we can check structural connectivity here.
    
    Parameters
    ----------
    graph_def : Dict[str, Any]
        The validated graph definition.
    agent_capabilities : Optional[Dict[str, Any]]
        Map of agent_type -> capability dict (optional).
    
    Returns
    -------
    List[str]
        List of warning messages (doesn't raise exception to allow flexibility).
    """
    warnings = []
    nodes = {n["id"]: n for n in graph_def.get("nodes", [])}
    edges = graph_def.get("edges", [])
    
    # Map inputs for each node
    node_inputs: Dict[str, Set[str]] = {nid: set() for nid in nodes}
    
    for edge in edges:
        to_node = edge["to"]
        mapping = edge.get("data_mapping", {})
        # If mapping is empty, it acts as a trigger or passes implicit generic data?
        # Usually data_mapping keys are inputs to the consumer.
        # Format: { "output_key_from_producer": "input_key_for_consumer" }
        for _, input_key in mapping.items():
            node_inputs[to_node].add(input_key)
            
    # Check against capabilities if provided
    if agent_capabilities:
        for node in graph_def.get("nodes", []):
            node_id = node["id"]
            node_type = node["type"]
            caps = agent_capabilities.get(node_type)
            
            if caps and "needs_input_keys" in caps:
                required = set(caps["needs_input_keys"])
                provided = node_inputs[node_id]
                missing = required - provided
                
                # Check for "optional" inputs or special flags like "consumes_all_outputs"
                # If consumes_all_outputs is true, it might not need explicit mappings
                # But if explicit inputs are required, they should be mapped.
                
                if missing:
                    warnings.append(
                        f"Node '{node_id}' (type {node_type}) is missing required inputs: {missing}"
                    )

    return warnings


def _detect_cycles(graph_def: Dict[str, Any]) -> None:
    """Detect if the graph contains any cycles using DFS."""
    nodes = graph_def.get("nodes", [])
    edges = graph_def.get("edges", [])
    
    # Build adjacency list
    adj: Dict[str, List[str]] = {n["id"]: [] for n in nodes}
    for edge in edges:
        from_node = edge["from"]
        to_node = edge["to"]
        if from_node in adj:
            adj[from_node].append(to_node)
            
    # DFS for cycle detection
    visited: Set[str] = set()
    recursion_stack: Set[str] = set()
    
    def visit(node_id: str):
        visited.add(node_id)
        recursion_stack.add(node_id)
        
        for neighbor in adj.get(node_id, []):
            if neighbor not in visited:
                visit(neighbor)
            elif neighbor in recursion_stack:
                raise GraphValidationError(f"Cycle detected involving node '{neighbor}'")
                
        recursion_stack.remove(node_id)
        
    for node in nodes:
        node_id = node["id"]
        if node_id not in visited:
            visit(node_id)
