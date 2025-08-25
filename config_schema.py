from typing import Any, Dict, List
from pydantic import BaseModel, Field, model_validator, ConfigDict


class NodeDefinition(BaseModel):
    id: str
    type: str
    config: Dict[str, Any] = Field(default_factory=dict)


class EdgeDefinition(BaseModel):
    from_node: str = Field(..., alias="from")
    to_node: str = Field(..., alias="to")
    data_mapping: Dict[str, str] = Field(default_factory=dict)

    model_config = ConfigDict(populate_by_name=True)


class GraphDefinition(BaseModel):
    nodes: List[NodeDefinition]
    edges: List[EdgeDefinition]

    @model_validator(mode="after")
    def check_edges_reference_known_nodes(self):
        node_ids = {node.id for node in self.nodes}
        for edge in self.edges:
            if edge.from_node not in node_ids or edge.to_node not in node_ids:
                raise ValueError(
                    f"Edge references undefined node id: {edge.from_node} -> {edge.to_node}"
                )
        return self


def validate_graph_definition(graph_def: Dict[str, Any]) -> Dict[str, Any]:
    """Validate and normalize a graph definition dictionary."""
    graph = GraphDefinition.model_validate(graph_def)
    return graph.model_dump(by_alias=True)
