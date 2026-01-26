"""Context injection utilities for agent execution.

This module provides utilities for injecting context and gathering inputs
for agent nodes based on their declared capabilities.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set

from contracts import AgentCapabilities, UpstreamIssue, InputGatherResult
from utils import log_status


# Agent types that require special context injection
AGENTS_REQUIRING_PROJECT_DIR = {
    "KnowledgeIntegratorAgent",
    "HypothesisGeneratorAgent",
    "ExperimentDesignerAgent",
    "ExperimentalDataLoaderAgent",
}

AGENTS_CONSUMING_ALL_OUTPUTS = {
    "ObserverAgent",
}


def get_agent_capabilities(agent_type: str, config_params: Dict[str, Any]) -> AgentCapabilities:
    """Determine agent capabilities from type and config.

    Parameters
    ----------
    agent_type : str
        The type name of the agent.
    config_params : Dict[str, Any]
        The agent's configuration parameters.

    Returns
    -------
    AgentCapabilities
        Declared capabilities for the agent.
    """
    return AgentCapabilities(
        requires_project_dir=agent_type in AGENTS_REQUIRING_PROJECT_DIR,
        consumes_all_outputs=agent_type in AGENTS_CONSUMING_ALL_OUTPUTS,
        loop_over=config_params.get("loop_over"),
        parallelizable=config_params.get("parallel_execution", False),
    )


def gather_inputs(
    node_id: str,
    incoming_edges: List[Dict[str, Any]],
    outputs_history: Dict[str, Dict[str, Any]],
    initial_inputs: Dict[str, Any],
    declared_output_keys: Dict[str, Set[str]],
    optional_inputs: Optional[Set[str]] = None,
) -> InputGatherResult:
    """Gather inputs for a node from upstream outputs and initial inputs.

    Parameters
    ----------
    node_id : str
        The ID of the node to gather inputs for.
    incoming_edges : List[Dict[str, Any]]
        List of edge definitions pointing to this node.
    outputs_history : Dict[str, Dict[str, Any]]
        Mapping of node IDs to their outputs.
    initial_inputs : Dict[str, Any]
        Initial workflow inputs.
    declared_output_keys : Dict[str, Set[str]]
        Mapping of node IDs to their declared output keys.
    optional_inputs : Optional[Set[str]]
        Set of input keys that are optional.

    Returns
    -------
    InputGatherResult
        Result containing gathered inputs and any issues.
    """
    if optional_inputs is None:
        optional_inputs = set()

    agent_inputs: Dict[str, Any] = {}
    issues: List[UpstreamIssue] = []

    for edge in incoming_edges:
        source_node = edge.get("from")
        data_mapping = edge.get("data_mapping", {})
        source_output = outputs_history.get(source_node, {})

        for src_key, dest_key in data_mapping.items():
            value, issue = _resolve_input_value(
                source_node=source_node,
                src_key=src_key,
                dest_key=dest_key,
                source_output=source_output,
                initial_inputs=initial_inputs,
                declared_output_keys=declared_output_keys,
                optional_inputs=optional_inputs,
            )

            if issue:
                issues.append(issue)

            if value is not None:
                agent_inputs[dest_key] = value

    return InputGatherResult(inputs=agent_inputs, issues=issues)


def _resolve_input_value(
    source_node: str,
    src_key: str,
    dest_key: str,
    source_output: Dict[str, Any],
    initial_inputs: Dict[str, Any],
    declared_output_keys: Dict[str, Set[str]],
    optional_inputs: Set[str],
) -> tuple[Any, Optional[UpstreamIssue]]:
    """Resolve a single input value from upstream output or initial inputs.

    Parameters
    ----------
    source_node : str
        The source node ID.
    src_key : str
        The key to read from source output.
    dest_key : str
        The key to write to in agent inputs.
    source_output : Dict[str, Any]
        The source node's output dictionary.
    initial_inputs : Dict[str, Any]
        Initial workflow inputs.
    declared_output_keys : Dict[str, Set[str]]
        Mapping of node IDs to their declared output keys.
    optional_inputs : Set[str]
        Set of optional input keys.

    Returns
    -------
    Tuple[Any, Optional[UpstreamIssue]]
        The resolved value (or None) and any issue encountered.
    """
    # Check for upstream error
    if source_output.get("error"):
        return None, UpstreamIssue(
            source_node=source_node,
            target_key=dest_key,
            message=f"Upstream error from '{source_node}': {source_output['error']}",
            fatal=True,
        )

    # Try to get value from source output
    if src_key in source_output:
        return source_output[src_key], None

    # Fallback to initial inputs
    if src_key in initial_inputs:
        return initial_inputs[src_key], None

    # Check if this is an optional input
    if dest_key in optional_inputs or src_key in optional_inputs:
        return None, None

    # Missing required input
    is_declared = src_key in declared_output_keys.get(source_node, set())
    return None, UpstreamIssue(
        source_node=source_node,
        target_key=dest_key,
        message=f"Key '{src_key}' not found in output from '{source_node}'"
                + (" (declared but missing)" if is_declared else ""),
        fatal=False,  # Continue with None value
    )


def inject_context(
    agent_inputs: Dict[str, Any],
    capabilities: AgentCapabilities,
    project_base_output_dir: Optional[str] = None,
    all_agent_outputs: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Inject additional context based on agent capabilities.

    Parameters
    ----------
    agent_inputs : Dict[str, Any]
        The base inputs gathered for the agent.
    capabilities : AgentCapabilities
        The agent's declared capabilities.
    project_base_output_dir : Optional[str]
        The project output directory path.
    all_agent_outputs : Optional[Dict[str, Dict[str, Any]]]
        All previous agent outputs.

    Returns
    -------
    Dict[str, Any]
        Enhanced inputs with injected context.
    """
    result = dict(agent_inputs)

    if capabilities.requires_project_dir and project_base_output_dir:
        result["project_base_output_dir"] = project_base_output_dir

    if capabilities.consumes_all_outputs and all_agent_outputs:
        result["all_agent_outputs"] = all_agent_outputs

    return result


def append_upstream_issue(
    agent_inputs: Dict[str, Any],
    issue: UpstreamIssue,
) -> None:
    """Record an upstream issue in the agent inputs.

    This maintains backward compatibility with the existing error
    aggregation format used by the orchestrator.

    Parameters
    ----------
    agent_inputs : Dict[str, Any]
        The agent inputs dictionary to update.
    issue : UpstreamIssue
        The upstream issue to record.
    """
    if not issue.message:
        return

    detail_entry = {
        "source": issue.source_node,
        "target": issue.target_key,
        "message": issue.message,
    }
    agent_inputs.setdefault("upstream_error_details", []).append(detail_entry)

    messages_list = agent_inputs.setdefault("upstream_error_messages", [])
    if issue.message not in messages_list:
        messages_list.append(issue.message)

    if issue.fatal:
        agent_inputs["error"] = "; ".join(messages_list)
