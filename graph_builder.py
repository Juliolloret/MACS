"""Graph construction and topological ordering utilities."""

from __future__ import annotations

from collections import defaultdict, deque
from typing import Any, Dict, List, Set, Tuple


class GraphBuildError(Exception):
    """Raised when graph construction fails."""


def build_adjacency_structures(
    graph_definition: Dict[str, Any],
    exclude_node_ids: Set[str] | None = None,
) -> Tuple[Dict[str, List[str]], Dict[str, List[Dict[str, Any]]], Dict[str, Set[str]]]:
    """Build adjacency lists and edge mappings from a graph definition.

    Parameters
    ----------
    graph_definition : Dict[str, Any]
        Graph definition containing 'nodes' and 'edges' keys.
    exclude_node_ids : Set[str] | None
        Node IDs to exclude from the graph (e.g., 'observer').

    Returns
    -------
    Tuple containing:
        - adjacency_list: Dict mapping source node ID to list of target node IDs
        - incoming_edges_map: Dict mapping target node ID to list of edge definitions
        - declared_output_keys: Dict mapping node ID to set of output keys declared in edges

    Raises
    ------
    GraphBuildError
        If an edge references an undefined node.
    """
    if exclude_node_ids is None:
        exclude_node_ids = set()

    nodes = graph_definition.get("nodes", [])
    edges = graph_definition.get("edges", [])

    # Build set of all node IDs for validation
    all_node_ids = {node["id"] for node in nodes}
    # Node IDs excluding special nodes
    active_node_ids = {node["id"] for node in nodes if node["id"] not in exclude_node_ids}

    adjacency_list: Dict[str, List[str]] = defaultdict(list)
    incoming_edges_map: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    declared_output_keys: Dict[str, Set[str]] = defaultdict(set)

    for edge in edges:
        from_node = edge.get("from")
        to_node = edge.get("to")

        # Validate edge endpoints exist
        if from_node not in all_node_ids or to_node not in all_node_ids:
            raise GraphBuildError(
                f"Edge references undefined node ID: {from_node} -> {to_node}"
            )

        # Build incoming edges map (always, for input gathering)
        incoming_edges_map[to_node].append(edge)

        # Build adjacency list for nodes we're sorting
        if from_node in active_node_ids and to_node in active_node_ids:
            adjacency_list[from_node].append(to_node)

        # Track declared output keys from data mappings
        data_mapping = edge.get("data_mapping", {})
        for src_key in data_mapping.keys():
            if from_node:
                declared_output_keys[from_node].add(src_key)

    return dict(adjacency_list), dict(incoming_edges_map), dict(declared_output_keys)


def compute_topological_order(
    graph_definition: Dict[str, Any],
    exclude_node_ids: Set[str] | None = None,
) -> List[str]:
    """Compute a topological ordering of nodes in the graph.

    Uses Kahn's algorithm to determine a valid execution order where
    each node is executed only after all its dependencies.

    Parameters
    ----------
    graph_definition : Dict[str, Any]
        Graph definition containing 'nodes' and 'edges' keys.
    exclude_node_ids : Set[str] | None
        Node IDs to exclude from ordering (e.g., 'observer').

    Returns
    -------
    List[str]
        List of node IDs in topological order.

    Raises
    ------
    GraphBuildError
        If the graph contains a cycle or is disconnected.
    """
    if exclude_node_ids is None:
        exclude_node_ids = set()

    nodes = graph_definition.get("nodes", [])
    edges = graph_definition.get("edges", [])

    # Build set of active node IDs
    node_ids = {node["id"] for node in nodes if node["id"] not in exclude_node_ids}

    if not node_ids:
        return []

    # Initialize in-degree for each node
    in_degree = {node_id: 0 for node_id in node_ids}

    # Build adjacency list and compute in-degrees
    adjacency_list: Dict[str, List[str]] = defaultdict(list)
    for edge in edges:
        from_node = edge.get("from")
        to_node = edge.get("to")

        if from_node in node_ids and to_node in node_ids:
            adjacency_list[from_node].append(to_node)
            in_degree[to_node] += 1

    # Kahn's algorithm
    queue = deque([node_id for node_id in node_ids if in_degree[node_id] == 0])
    ordered: List[str] = []

    while queue:
        node = queue.popleft()
        ordered.append(node)

        for neighbor in adjacency_list.get(node, []):
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    # Check for cycles
    if len(ordered) != len(node_ids):
        processed = set(ordered)
        missing = node_ids - processed
        raise GraphBuildError(
            f"Graph has a cycle or is disconnected. "
            f"Processed {len(ordered)}/{len(node_ids)} nodes. "
            f"Problematic nodes: {missing}"
        )

    return ordered


def get_node_by_id(graph_definition: Dict[str, Any], node_id: str) -> Dict[str, Any] | None:
    """Look up a node definition by its ID.

    Parameters
    ----------
    graph_definition : Dict[str, Any]
        Graph definition containing 'nodes' key.
    node_id : str
        The node identifier to look up.

    Returns
    -------
    Dict[str, Any] | None
        The node definition dictionary, or None if not found.
    """
    for node in graph_definition.get("nodes", []):
        if node.get("id") == node_id:
            return node
    return None


def get_node_type(graph_definition: Dict[str, Any], node_id: str) -> str | None:
    """Get the type of a node by its ID.

    Parameters
    ----------
    graph_definition : Dict[str, Any]
        Graph definition containing 'nodes' key.
    node_id : str
        The node identifier to look up.

    Returns
    -------
    str | None
        The node type, or None if node not found.
    """
    node = get_node_by_id(graph_definition, node_id)
    return node.get("type") if node else None


def get_node_config(graph_definition: Dict[str, Any], node_id: str) -> Dict[str, Any]:
    """Get the config dictionary for a node.

    Parameters
    ----------
    graph_definition : Dict[str, Any]
        Graph definition containing 'nodes' key.
    node_id : str
        The node identifier to look up.

    Returns
    -------
    Dict[str, Any]
        The node config dictionary, or empty dict if node not found.
    """
    node = get_node_by_id(graph_definition, node_id)
    return node.get("config", {}) if node else {}
