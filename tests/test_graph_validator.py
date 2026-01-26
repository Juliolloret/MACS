"""Tests for graph_validator module."""

import pytest
from graph_validator import validate_graph_structure, GraphValidationError, validate_node_inputs

class TestGraphValidator:
    
    def test_valid_dag(self):
        """Test a valid DAG structure."""
        graph_def = {
            "nodes": [
                {"id": "A", "type": "StartAgent"},
                {"id": "B", "type": "ProcessAgent"},
                {"id": "C", "type": "EndAgent"}
            ],
            "edges": [
                {"from": "A", "to": "B"},
                {"from": "B", "to": "C"}
            ]
        }
        validated = validate_graph_structure(graph_def)
        assert validated["nodes"][0]["id"] == "A"

    def test_cycle_detection(self):
        """Test that cycles are detected."""
        graph_def = {
            "nodes": [
                {"id": "A", "type": "Agent"},
                {"id": "B", "type": "Agent"}
            ],
            "edges": [
                {"from": "A", "to": "B"},
                {"from": "B", "to": "A"}  # Cycle
            ]
        }
        with pytest.raises(GraphValidationError, match="Cycle detected"):
            validate_graph_structure(graph_def)

    def test_self_cycle_detection(self):
        """Test that self-cycles are detected."""
        graph_def = {
            "nodes": [
                {"id": "A", "type": "Agent"}
            ],
            "edges": [
                {"from": "A", "to": "A"}
            ]
        }
        with pytest.raises(GraphValidationError, match="Cycle detected"):
            validate_graph_structure(graph_def)

    def test_schema_validation_failure(self):
        """Test that schema violations are caught."""
        graph_def = {
            "nodes": [
                {"id": "A"}  # Missing type
            ],
            "edges": []
        }
        with pytest.raises(GraphValidationError, match="Schema validation failed"):
            validate_graph_structure(graph_def)
            
    def test_invalid_node_reference(self):
        """Test that edges referencing unknown nodes are caught."""
        graph_def = {
            "nodes": [{"id": "A", "type": "Agent"}],
            "edges": [{"from": "A", "to": "Z"}]  # Z does not exist
        }
        # This comes from config_schema pydantic validator
        with pytest.raises(GraphValidationError, match="undefined node id"):
            validate_graph_structure(graph_def)

class TestNodeInputValidation:
    
    def test_validate_node_inputs_missing(self):
        """Test detection of missing required inputs."""
        graph_def = {
            "nodes": [
                {"id": "A", "type": "Producer"},
                {"id": "B", "type": "Consumer"}
            ],
            "edges": [
                {"from": "A", "to": "B", "data_mapping": {}} # No mapping
            ]
        }
        
        caps = {
            "Consumer": {"needs_input_keys": ["input_data"]}
        }
        
        warnings = validate_node_inputs(graph_def, caps)
        assert len(warnings) == 1
        assert "Node 'B'" in warnings[0]
        assert "missing required inputs" in warnings[0]
        assert "input_data" in warnings[0]

    def test_validate_node_inputs_satisfied(self):
        """Test that mapped inputs satisfy requirements."""
        graph_def = {
            "nodes": [
                {"id": "A", "type": "Producer"},
                {"id": "B", "type": "Consumer"}
            ],
            "edges": [
                {
                    "from": "A", 
                    "to": "B", 
                    "data_mapping": {"output": "input_data"}
                }
            ]
        }
        
        caps = {
            "Consumer": {"needs_input_keys": ["input_data"]}
        }
        
        warnings = validate_node_inputs(graph_def, caps)
        assert len(warnings) == 0
