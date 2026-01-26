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


class TestGraphValidationResult:
    """Tests for the GraphValidationResult dataclass."""

    def test_default_is_valid(self):
        """Test that a new result starts as valid."""
        from graph_validator import GraphValidationResult
        result = GraphValidationResult()
        assert result.valid is True
        assert result.errors == []
        assert result.warnings == []

    def test_add_error_marks_invalid(self):
        """Test that adding an error marks the result as invalid."""
        from graph_validator import GraphValidationResult
        result = GraphValidationResult()
        result.add_error("Test error")
        assert result.valid is False
        assert "Test error" in result.errors

    def test_add_warning_keeps_valid(self):
        """Test that adding a warning doesn't change validity."""
        from graph_validator import GraphValidationResult
        result = GraphValidationResult()
        result.add_warning("Test warning")
        assert result.valid is True
        assert "Test warning" in result.warnings


class TestValidateAgentTypes:
    """Tests for the validate_agent_types function."""

    def test_valid_agent_types(self):
        """Test that known agent types pass validation."""
        from graph_validator import validate_agent_types
        graph_def = {
            "nodes": [
                {"id": "A", "type": "TestAgent"},
            ],
            "edges": []
        }
        mock_registry = {"TestAgent": {"class": object}}
        errors = validate_agent_types(graph_def, mock_registry)
        assert len(errors) == 0

    def test_unknown_agent_type(self):
        """Test that unknown agent types generate errors."""
        from graph_validator import validate_agent_types
        graph_def = {
            "nodes": [
                {"id": "A", "type": "UnknownAgent"},
            ],
            "edges": []
        }
        mock_registry = {"KnownAgent": {"class": object}}
        errors = validate_agent_types(graph_def, mock_registry)
        assert len(errors) == 1
        assert "UnknownAgent" in errors[0]
        assert "Node 'A'" in errors[0]


class TestValidateDataMappings:
    """Tests for the validate_data_mappings function."""

    def test_valid_mappings(self):
        """Test that valid mappings pass validation."""
        from graph_validator import validate_data_mappings
        graph_def = {
            "nodes": [
                {"id": "A", "type": "Producer"},
                {"id": "B", "type": "Consumer"}
            ],
            "edges": [
                {"from": "A", "to": "B", "data_mapping": {"output": "input"}}
            ]
        }
        errors, warnings = validate_data_mappings(graph_def)
        assert len(errors) == 0

    def test_invalid_mapping_value_type(self):
        """Test that invalid mapping values generate errors."""
        from graph_validator import validate_data_mappings
        graph_def = {
            "nodes": [
                {"id": "A", "type": "Producer"},
                {"id": "B", "type": "Consumer"}
            ],
            "edges": [
                {"from": "A", "to": "B", "data_mapping": {"output": None}}
            ]
        }
        errors, warnings = validate_data_mappings(graph_def)
        assert len(errors) == 1
        assert "invalid mapping value" in errors[0]

    def test_orphaned_node_warning(self):
        """Test that orphaned nodes generate warnings."""
        from graph_validator import validate_data_mappings
        graph_def = {
            "nodes": [
                {"id": "A", "type": "Standalone"},
                {"id": "B", "type": "Producer"},
                {"id": "C", "type": "Consumer"}
            ],
            "edges": [
                {"from": "B", "to": "C", "data_mapping": {}}
            ]
        }
        errors, warnings = validate_data_mappings(graph_def)
        assert len(errors) == 0
        # Node A is orphaned
        assert any("Node 'A'" in w for w in warnings)


class TestValidateGraphComprehensive:
    """Tests for the comprehensive validation function."""

    def test_valid_graph_comprehensive(self):
        """Test that a valid graph passes comprehensive validation."""
        from graph_validator import validate_graph_comprehensive
        graph_def = {
            "nodes": [
                {"id": "A", "type": "StartAgent"},
                {"id": "B", "type": "EndAgent"}
            ],
            "edges": [
                {"from": "A", "to": "B", "data_mapping": {"output": "input"}}
            ]
        }
        result = validate_graph_comprehensive(graph_def, check_agent_types=False)
        assert result.valid is True
        assert result.validated_graph is not None

    def test_schema_error_returns_early(self):
        """Test that schema errors prevent further validation."""
        from graph_validator import validate_graph_comprehensive
        graph_def = {
            "nodes": [{"id": "A"}],  # Missing type
            "edges": []
        }
        result = validate_graph_comprehensive(graph_def, check_agent_types=False)
        assert result.valid is False
        assert any("Schema validation failed" in e for e in result.errors)

    def test_cycle_error(self):
        """Test that cycles are detected in comprehensive validation."""
        from graph_validator import validate_graph_comprehensive
        graph_def = {
            "nodes": [
                {"id": "A", "type": "Agent"},
                {"id": "B", "type": "Agent"}
            ],
            "edges": [
                {"from": "A", "to": "B"},
                {"from": "B", "to": "A"}
            ]
        }
        result = validate_graph_comprehensive(graph_def, check_agent_types=False)
        assert result.valid is False
        assert any("Cycle" in e for e in result.errors)

    def test_unknown_agent_type_error(self):
        """Test that unknown agent types are reported as errors."""
        from graph_validator import validate_graph_comprehensive
        graph_def = {
            "nodes": [{"id": "A", "type": "UnknownAgent"}],
            "edges": []
        }
        mock_registry = {"KnownAgent": {"class": object}}
        result = validate_graph_comprehensive(
            graph_def,
            agent_registry=mock_registry,
            check_agent_types=True
        )
        assert result.valid is False
        assert any("UnknownAgent" in e for e in result.errors)

    def test_missing_input_warning(self):
        """Test that missing required inputs generate warnings."""
        from graph_validator import validate_graph_comprehensive
        graph_def = {
            "nodes": [
                {"id": "A", "type": "Producer"},
                {"id": "B", "type": "Consumer"}
            ],
            "edges": [
                {"from": "A", "to": "B", "data_mapping": {}}
            ]
        }
        caps = {"Consumer": {"needs_input_keys": ["required_input"]}}
        result = validate_graph_comprehensive(
            graph_def,
            agent_capabilities=caps,
            check_agent_types=False
        )
        # Missing inputs are warnings, not errors
        assert any("missing required inputs" in w for w in result.warnings)
