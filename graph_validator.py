"""Graph validation mechanisms including schema checks and cycle detection."""

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from config_schema import validate_graph_definition


class GraphValidationError(ValueError):
    """Raised when the graph structure is invalid."""
    pass


@dataclass
class GraphValidationResult:
    """Comprehensive validation result with errors and warnings.

    Attributes
    ----------
    valid : bool
        True if no fatal errors were found.
    errors : List[str]
        Fatal validation errors that prevent execution.
    warnings : List[str]
        Non-fatal issues that should be logged but don't block execution.
    validated_graph : Optional[Dict[str, Any]]
        The validated and normalized graph definition, or None if invalid.
    """

    valid: bool = True
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    validated_graph: Optional[Dict[str, Any]] = None

    def add_error(self, message: str) -> None:
        """Add a fatal error and mark result as invalid."""
        self.errors.append(message)
        self.valid = False

    def add_warning(self, message: str) -> None:
        """Add a non-fatal warning."""
        self.warnings.append(message)


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

                # If consumes_all_outputs is true, it might not need explicit mappings
                if caps.get("consumes_all_outputs"):
                    continue

                if missing:
                    warnings.append(
                        f"Node '{node_id}' (type {node_type}) is missing required inputs: {missing}"
                    )

    return warnings


def validate_agent_types(
    graph_def: Dict[str, Any],
    agent_registry: Optional[Dict[str, Any]] = None
) -> List[str]:
    """Validate that all node types are registered agents.

    Parameters
    ----------
    graph_def : Dict[str, Any]
        The graph definition with nodes.
    agent_registry : Optional[Dict[str, Any]]
        Agent registry mapping type names to classes. If None, uses global registry.

    Returns
    -------
    List[str]
        List of error messages for unregistered agent types.
    """
    errors = []

    if agent_registry is None:
        try:
            from agents.registry import AGENT_REGISTRY
            agent_registry = AGENT_REGISTRY
        except ImportError:
            # Registry not available, skip validation
            return errors

    registered_types = set(agent_registry.keys())

    for node in graph_def.get("nodes", []):
        node_type = node.get("type")
        if node_type and node_type not in registered_types:
            available = ", ".join(sorted(registered_types)[:10])
            suffix = "..." if len(registered_types) > 10 else ""
            errors.append(
                f"Node '{node['id']}' uses unknown agent type '{node_type}'. "
                f"Available types: [{available}{suffix}]. "
                f"Check that the agent is registered via @register_agent decorator."
            )

    return errors


def validate_data_mappings(
    graph_def: Dict[str, Any],
    agent_capabilities: Optional[Dict[str, Any]] = None
) -> Tuple[List[str], List[str]]:
    """Validate data_mapping keys are consistent with agent I/O.

    Checks that data_mapping source keys reference valid output keys from
    the producing node (when agent capabilities are available).

    Parameters
    ----------
    graph_def : Dict[str, Any]
        The validated graph definition.
    agent_capabilities : Optional[Dict[str, Any]]
        Map of agent_type -> capability dict with declared outputs.

    Returns
    -------
    Tuple[List[str], List[str]]
        (errors, warnings) tuple. Errors are fatal, warnings are informational.
    """
    errors: List[str] = []
    warnings: List[str] = []

    edges = graph_def.get("edges", [])
    nodes = {n["id"]: n for n in graph_def.get("nodes", [])}

    # Track what each node is expected to output based on downstream mappings
    expected_outputs: Dict[str, Set[str]] = defaultdict(set)

    for edge in edges:
        from_node = edge.get("from")
        to_node = edge.get("to")
        data_mapping = edge.get("data_mapping", {})

        for source_key, target_key in data_mapping.items():
            if from_node:
                expected_outputs[from_node].add(source_key)

            # Validate that target_key is a valid identifier
            if not target_key or not isinstance(target_key, str):
                errors.append(
                    f"Edge '{from_node}' -> '{to_node}' has invalid mapping value "
                    f"for key '{source_key}': expected string, got {type(target_key).__name__}"
                )

    # Check for orphaned nodes (no incoming or outgoing edges)
    node_ids = set(nodes.keys())
    connected_nodes: Set[str] = set()
    for edge in edges:
        connected_nodes.add(edge.get("from", ""))
        connected_nodes.add(edge.get("to", ""))

    orphaned = node_ids - connected_nodes
    for node_id in orphaned:
        # Skip observer which is intentionally disconnected
        if nodes.get(node_id, {}).get("type") == "ObserverAgent":
            continue
        warnings.append(
            f"Node '{node_id}' has no edges connecting it to other nodes."
        )

    return errors, warnings


def validate_graph_comprehensive(
    graph_def: Dict[str, Any],
    agent_registry: Optional[Dict[str, Any]] = None,
    agent_capabilities: Optional[Dict[str, Any]] = None,
    check_agent_types: bool = True
) -> GraphValidationResult:
    """Comprehensive graph validation with structured results.

    Runs all validation checks and collects errors and warnings separately.
    This is the recommended entry point for full validation.

    Parameters
    ----------
    graph_def : Dict[str, Any]
        Raw graph definition to validate.
    agent_registry : Optional[Dict[str, Any]]
        Agent registry for type validation. Uses global if None.
    agent_capabilities : Optional[Dict[str, Any]]
        Agent capabilities for input/output validation.
    check_agent_types : bool
        Whether to validate agent types against registry.

    Returns
    -------
    GraphValidationResult
        Structured result with errors, warnings, and validated graph.

    Examples
    --------
    >>> result = validate_graph_comprehensive(graph_def, check_agent_types=False)
    >>> if result.valid:
    ...     orchestrator.run(result.validated_graph)
    ... else:
    ...     for error in result.errors:
    ...         print(f"ERROR: {error}")
    """
    result = GraphValidationResult()

    # 1. Schema validation
    try:
        validated_graph = validate_graph_definition(graph_def)
        result.validated_graph = validated_graph
    except ValueError as e:
        result.add_error(f"Schema validation failed: {e}")
        return result

    # 2. Cycle detection
    try:
        _detect_cycles(validated_graph)
    except GraphValidationError as e:
        result.add_error(str(e))
        return result

    # 3. Agent type validation (optional, for early error detection)
    if check_agent_types:
        type_errors = validate_agent_types(validated_graph, agent_registry)
        for err in type_errors:
            result.add_error(err)

    # 4. Data mapping validation
    mapping_errors, mapping_warnings = validate_data_mappings(
        validated_graph, agent_capabilities
    )
    for err in mapping_errors:
        result.add_error(err)
    for warn in mapping_warnings:
        result.add_warning(warn)

    # 5. Node input validation
    input_warnings = validate_node_inputs(validated_graph, agent_capabilities)
    for warn in input_warnings:
        result.add_warning(warn)

    return result


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
