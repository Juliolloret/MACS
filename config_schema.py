"""Pydantic models describing the orchestration graph schema."""

from typing import Any, Dict, List

try:  # Support both Pydantic v1 and v2
    from pydantic import BaseModel, Field, model_validator, ConfigDict
    PYDANTIC_V2 = True
except ImportError:  # pragma: no cover - exercised in older environments
    from pydantic import BaseModel, Field, root_validator

    PYDANTIC_V2 = False

    def model_validator(*_args, **_kwargs):
        """Shim for v1 root validators to match v2 decorator usage."""
        def decorator(func):
            return root_validator(pre=False, allow_reuse=True)(func)
        return decorator

    class ConfigDict(dict):
        """Shim for v1 compatibility where ConfigDict is not available."""

        def __init__(self, **kwargs):
            super().__init__(**kwargs)


class NodeDefinition(BaseModel):  # pylint: disable=too-few-public-methods
    """Schema describing a single node within the orchestration graph."""

    id: str
    type: str
    config: Dict[str, Any] = Field(default_factory=dict)


class EdgeDefinition(BaseModel):  # pylint: disable=too-few-public-methods
    """Schema describing a connection between two nodes."""

    from_node: str = Field(..., alias="from")
    to_node: str = Field(..., alias="to")
    data_mapping: Dict[str, str] = Field(default_factory=dict)

    if PYDANTIC_V2:
        model_config = ConfigDict(populate_by_name=True)
    else:
        class Config:  # pylint: disable=too-few-public-methods
            allow_population_by_field_name = True


class GraphDefinition(BaseModel):  # pylint: disable=too-few-public-methods
    """Container holding the full graph description."""

    nodes: List[NodeDefinition]
    edges: List[EdgeDefinition]

    if PYDANTIC_V2:
        @model_validator(mode="after")
        def check_edges_reference_known_nodes(self):
            """Ensure that every edge references existing node identifiers."""
            node_ids = {node.id for node in self.nodes}
            for edge in self.edges:
                if edge.from_node not in node_ids or edge.to_node not in node_ids:
                    raise ValueError(
                        f"Edge references undefined node id: {edge.from_node} -> {edge.to_node}"
                    )
            return self
    else:
        @model_validator()
        def check_edges_reference_known_nodes(cls, values):  # type: ignore[no-redef]
            """Ensure that every edge references existing node identifiers."""
            nodes = values.get("nodes", [])
            edges = values.get("edges", [])
            node_ids = {node.id for node in nodes}
            for edge in edges:
                if edge.from_node not in node_ids or edge.to_node not in node_ids:
                    raise ValueError(
                        f"Edge references undefined node id: {edge.from_node} -> {edge.to_node}"
                    )
            return values


def validate_graph_definition(graph_def: Dict[str, Any]) -> Dict[str, Any]:
    """Validate and normalize a graph definition dictionary."""
    if PYDANTIC_V2:
        graph = GraphDefinition.model_validate(graph_def)
        return graph.model_dump(by_alias=True)
    graph = GraphDefinition.parse_obj(graph_def)
    return graph.dict(by_alias=True)
