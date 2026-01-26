"""Tests for graph_builder module."""

import pytest

from graph_builder import (
    build_adjacency_structures,
    compute_topological_order,
    get_node_by_id,
    get_node_type,
    get_node_config,
    GraphBuildError,
)


class TestBuildAdjacencyStructures:
    """Tests for adjacency structure building."""

    def test_build_simple_graph(self):
        """Test building structures for a simple linear graph."""
        graph_def = {
            "nodes": [
                {"id": "a", "type": "TypeA"},
                {"id": "b", "type": "TypeB"},
            ],
            "edges": [
                {"from": "a", "to": "b", "data_mapping": {"out": "in"}}
            ]
        }

        adj_list, incoming, declared = build_adjacency_structures(graph_def)

        assert adj_list == {"a": ["b"]}
        assert "b" in incoming
        assert "out" in declared["a"]

    def test_exclude_nodes(self):
        """Test that excluded nodes are not in adjacency list."""
        graph_def = {
            "nodes": [
                {"id": "a", "type": "TypeA"},
                {"id": "observer", "type": "Observer"},
            ],
            "edges": []
        }

        adj_list, _, _ = build_adjacency_structures(
            graph_def, exclude_node_ids={"observer"}
        )

        assert "observer" not in adj_list

    def test_undefined_node_in_edge_raises(self):
        """Test that edge referencing undefined node raises error."""
        graph_def = {
            "nodes": [{"id": "a", "type": "TypeA"}],
            "edges": [{"from": "a", "to": "undefined"}]
        }

        with pytest.raises(GraphBuildError, match="undefined"):
            build_adjacency_structures(graph_def)


class TestComputeTopologicalOrder:
    """Tests for topological ordering."""

    def test_simple_linear_order(self):
        """Test ordering of a simple linear graph."""
        graph_def = {
            "nodes": [
                {"id": "a", "type": "TypeA"},
                {"id": "b", "type": "TypeB"},
                {"id": "c", "type": "TypeC"},
            ],
            "edges": [
                {"from": "a", "to": "b"},
                {"from": "b", "to": "c"},
            ]
        }

        order = compute_topological_order(graph_def)

        assert order.index("a") < order.index("b")
        assert order.index("b") < order.index("c")

    def test_diamond_graph(self):
        """Test ordering of a diamond-shaped graph."""
        graph_def = {
            "nodes": [
                {"id": "a", "type": "TypeA"},
                {"id": "b", "type": "TypeB"},
                {"id": "c", "type": "TypeC"},
                {"id": "d", "type": "TypeD"},
            ],
            "edges": [
                {"from": "a", "to": "b"},
                {"from": "a", "to": "c"},
                {"from": "b", "to": "d"},
                {"from": "c", "to": "d"},
            ]
        }

        order = compute_topological_order(graph_def)

        assert order.index("a") < order.index("b")
        assert order.index("a") < order.index("c")
        assert order.index("b") < order.index("d")
        assert order.index("c") < order.index("d")

    def test_cycle_detection(self):
        """Test that cycles are detected."""
        graph_def = {
            "nodes": [
                {"id": "a", "type": "TypeA"},
                {"id": "b", "type": "TypeB"},
            ],
            "edges": [
                {"from": "a", "to": "b"},
                {"from": "b", "to": "a"},
            ]
        }

        with pytest.raises(GraphBuildError, match="cycle"):
            compute_topological_order(graph_def)

    def test_empty_graph(self):
        """Test ordering of an empty graph."""
        graph_def = {"nodes": [], "edges": []}

        order = compute_topological_order(graph_def)

        assert order == []

    def test_exclude_observer(self):
        """Test that observer node is excluded from order."""
        graph_def = {
            "nodes": [
                {"id": "a", "type": "TypeA"},
                {"id": "observer", "type": "Observer"},
            ],
            "edges": []
        }

        order = compute_topological_order(graph_def, exclude_node_ids={"observer"})

        assert "a" in order
        assert "observer" not in order


class TestNodeLookup:
    """Tests for node lookup functions."""

    def test_get_node_by_id_found(self):
        """Test looking up an existing node."""
        graph_def = {
            "nodes": [
                {"id": "test_node", "type": "TestType", "config": {"key": "value"}}
            ]
        }

        node = get_node_by_id(graph_def, "test_node")

        assert node is not None
        assert node["id"] == "test_node"
        assert node["type"] == "TestType"

    def test_get_node_by_id_not_found(self):
        """Test looking up a non-existent node."""
        graph_def = {"nodes": []}

        node = get_node_by_id(graph_def, "missing")

        assert node is None

    def test_get_node_type(self):
        """Test getting node type."""
        graph_def = {
            "nodes": [{"id": "node1", "type": "MyType"}]
        }

        node_type = get_node_type(graph_def, "node1")

        assert node_type == "MyType"

    def test_get_node_type_not_found(self):
        """Test getting type of non-existent node."""
        graph_def = {"nodes": []}

        node_type = get_node_type(graph_def, "missing")

        assert node_type is None

    def test_get_node_config(self):
        """Test getting node config."""
        graph_def = {
            "nodes": [
                {"id": "node1", "type": "Type", "config": {"setting": True}}
            ]
        }

        config = get_node_config(graph_def, "node1")

        assert config == {"setting": True}

    def test_get_node_config_missing(self):
        """Test getting config of non-existent node."""
        graph_def = {"nodes": []}

        config = get_node_config(graph_def, "missing")

        assert config == {}
