"""Typed contracts for agent communication and orchestration."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class NodeContext:
    """Context information passed to agents during execution.

    This provides agents with execution metadata without requiring
    them to parse it from raw dictionaries.

    Attributes
    ----------
    run_id : str
        Unique identifier for the current workflow run.
    node_id : str
        Identifier of the current node being executed.
    attempt : int
        Current execution attempt (1-indexed). Greater than 1 for retries.
    project_base_output_dir : str
        Base directory where output artifacts should be written.
    requires_project_dir : bool
        Whether this agent needs the project directory injected.
    consumes_all_outputs : bool
        Whether this agent needs all previous agent outputs.
    """

    run_id: str
    node_id: str
    attempt: int
    project_base_output_dir: str
    requires_project_dir: bool = False
    consumes_all_outputs: bool = False


@dataclass
class NodeResult:
    """Standardized result returned by agent execution.

    Provides a structured way to return data, errors, metrics, and
    artifact paths from agent execution.

    Attributes
    ----------
    data : Dict[str, Any]
        Primary output data from the agent, keyed by output names.
    errors : List[str]
        List of error messages encountered during execution.
    warnings : List[str]
        List of warning messages (non-fatal issues).
    metrics : Dict[str, Any]
        Metrics collected during execution (timing, tokens, etc.).
    artifacts : List[str]
        Paths to any files created by the agent.
    """

    data: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    artifacts: List[str] = field(default_factory=list)

    @property
    def has_error(self) -> bool:
        """Return True if any errors were recorded."""
        return len(self.errors) > 0

    @property
    def error(self) -> Optional[str]:
        """Return combined error string for backward compatibility."""
        if not self.errors:
            return None
        return "; ".join(self.errors)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format for backward compatibility.

        Returns a dict that matches the legacy agent output format,
        with 'error' key if errors exist.
        """
        result = dict(self.data)
        if self.errors:
            result["error"] = self.error
        if self.warnings:
            result["warnings"] = list(self.warnings)
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "NodeResult":
        """Create a NodeResult from a legacy dictionary output.

        Parameters
        ----------
        data : Dict[str, Any]
            Dictionary with agent output, potentially containing 'error' key.

        Returns
        -------
        NodeResult
            Structured result object.
        """
        errors = []
        warnings = []
        output_data = {}

        for key, value in data.items():
            if key == "error" and value:
                errors.append(str(value))
            elif key == "warnings":
                if isinstance(value, list):
                    warnings.extend(value)
                elif value:
                    warnings.append(str(value))
            else:
                output_data[key] = value

        return cls(data=output_data, errors=errors, warnings=warnings)


@dataclass
class AgentCapabilities:
    """Declaration of agent capabilities and requirements.

    Agents can declare their capabilities so the orchestrator can
    inject appropriate context without special-casing by type name.

    Attributes
    ----------
    requires_project_dir : bool
        Agent needs project_base_output_dir injected.
    consumes_all_outputs : bool
        Agent needs all_agent_outputs with all prior results.
    needs_input_keys : List[str]
        Required input keys that must be provided.
    optional_input_keys : List[str]
        Optional input keys that may be provided.
    loop_over : Optional[str]
        If set, agent loops over this input key.
    parallelizable : bool
        Whether loop iterations can run in parallel.
    """

    requires_project_dir: bool = False
    consumes_all_outputs: bool = False
    needs_input_keys: List[str] = field(default_factory=list)
    optional_input_keys: List[str] = field(default_factory=list)
    loop_over: Optional[str] = None
    parallelizable: bool = False

    @classmethod
    def default(cls) -> "AgentCapabilities":
        """Return default capabilities for agents that don't declare them."""
        return cls()


@dataclass
class UpstreamIssue:
    """Record of an issue from an upstream node.

    Attributes
    ----------
    source_node : str
        ID of the node that produced the issue.
    target_key : str
        The input key that was affected.
    message : str
        Description of the issue.
    fatal : bool
        Whether this issue should prevent execution.
    """

    source_node: str
    target_key: str
    message: str
    fatal: bool = False


@dataclass
class InputGatherResult:
    """Result of gathering inputs for a node.

    Attributes
    ----------
    inputs : Dict[str, Any]
        The gathered input values.
    issues : List[UpstreamIssue]
        Any issues encountered during input gathering.
    has_fatal_error : bool
        Whether a fatal error prevents execution.
    """

    inputs: Dict[str, Any] = field(default_factory=dict)
    issues: List[UpstreamIssue] = field(default_factory=list)

    @property
    def has_fatal_error(self) -> bool:
        """Return True if any issue is marked as fatal."""
        return any(issue.fatal for issue in self.issues)

    @property
    def error_messages(self) -> List[str]:
        """Return list of all issue messages."""
        return [issue.message for issue in self.issues]
