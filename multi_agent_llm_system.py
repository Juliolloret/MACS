"""Graph orchestration utilities for coordinating multiple agents."""

from __future__ import annotations

import copy
import os
import time
import json
from collections import defaultdict, deque
import traceback
import threading
import webbrowser
from typing import Dict, Any, Optional, Set, Tuple, Union
from concurrent.futures import ThreadPoolExecutor, as_completed

from llm import LLMClient

try:
    from pyvis.network import Network
except ImportError:  # pragma: no cover - handled gracefully at runtime
    Network = None  # type: ignore

# Import utilities from utils.py
from utils import (
    log_status,
    set_status_callback,
    load_app_config,  # pylint: disable=unused-import
    set_graph_callback,  # re-exported for GUI usage
    report_graph_visualization,
    graph_callback_available,
)

# Configuration schema validation
from config_schema import validate_graph_definition

# New modular components
from graph_builder import (
    build_adjacency_structures,
    compute_topological_order,
    GraphBuildError,
)
from llm_factory import create_llm_client
from artifact_writer import ArtifactWriter, write_consolidated_outputs
from execution_engine import NodeExecutor
from context_injection import (
    get_agent_capabilities,
    gather_inputs,
    inject_context,
    append_upstream_issue,
)

# Run history utilities
from storage.run_history import generate_run_id, save_run

# Event bus utilities
from events import (
    EventBus,
    EVENT_GRAPH_VISUALIZED,
    EVENT_NODE_COMPLETED,
    EVENT_NODE_FAILED,
    EVENT_NODE_RETRY,
    EVENT_NODE_SKIPPED,
    EVENT_NODE_STARTED,
    EVENT_WORKFLOW_COMPLETED,
    EVENT_WORKFLOW_STARTED,
    NoopEventBus,
)
from event_subscribers import EventLogger, MetricsCollector

# Imports for refactored Agent classes
# Core agent utilities
from agents import get_agent_class
from agents.experimental_data_loader_agent import ExperimentalDataLoaderAgent
from agents.knowledge_integrator_agent import KnowledgeIntegratorAgent
# SDK Models are now in agents.sdk_models; WebResearcherAgent imports them directly.


# Script's Directory - this is specific to the main script file.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# SDK availability is now checked and logged in utils.py

# --- All class definitions (Agent, Pydantic models, specific agent classes) and utility functions are now removed from this file. ---
# They are imported from their respective locations in 'utils.py' or the 'agents' package.


def _open_graph_file(path: str):
    """Surface the rendered graph using the GUI callback when available."""

    if not path:
        return

    report_graph_visualization(path)

    if graph_callback_available():
        # The GUI listens for the callback and renders the graph inline.
        return

    def _viewer():
        try:
            webbrowser.open(f"file://{os.path.abspath(path)}")
        except Exception as exc:  # pragma: no cover - best effort only
            log_status(f"[GraphOrchestrator] WARNING: Could not open graph viewer: {exc}")

    threading.Thread(target=_viewer, daemon=True).start()

class GraphOrchestrator:
    """Execute a graph of agents according to a configuration definition."""

    def __init__(
        self,
        graph_definition_from_config,
        llm: LLMClient,
        app_config: Dict[str, Any],
        failure_policy: str = "continue",
        event_bus=None,
    ):
        """Create a new orchestrator.

        Parameters
        ----------
        graph_definition_from_config : dict
            Graph definition describing nodes and edges.
        llm : LLMClient
            Language model client used by agents.
        app_config : Dict[str, Any]
            Global application configuration.
        failure_policy : str, optional
            Default failure handling policy. One of ``"continue"``,
            ``"abort"`` or ``"retry"``. Defaults to ``"continue"``.
        """
        # Validate and normalize the graph definition before proceeding
        self.graph_definition = validate_graph_definition(graph_definition_from_config)
        self.app_config = app_config
        self.failure_policy = failure_policy
        self.agents = {}
        self.adjacency_list = defaultdict(list)
        self.incoming_edges_map = defaultdict(list)
        self.declared_output_keys = defaultdict(set)
        self.node_order = []
        self.llm = llm
        self.node_metrics = {}
        self.event_bus = event_bus if event_bus is not None else NoopEventBus()
        optional_inputs_from_config = (
            app_config.get("system_variables", {}).get("optional_initial_inputs")
            if isinstance(app_config, dict)
            else None
        )
        optional_inputs: Set[str] = {"user_query"}
        if isinstance(optional_inputs_from_config, (list, tuple, set)):
            for value in optional_inputs_from_config:
                if isinstance(value, str) and value:
                    optional_inputs.add(value)
        self.optional_initial_inputs = optional_inputs
        self._build_graph_and_determine_order()
        self._initialize_agents()

    def _build_graph_and_determine_order(self):
        """Build internal graph structures and determine execution order.

        Uses the graph_builder module for core logic while maintaining
        backward-compatible instance variables.
        """
        if 'nodes' not in self.graph_definition or 'edges' not in self.graph_definition:
            log_status("[GraphOrchestrator] ERROR: Graph definition in config must contain 'nodes' and 'edges'.")
            raise ValueError("Graph definition in config must contain 'nodes' and 'edges'.")

        # Use graph_builder module for adjacency structures
        try:
            adj_list, incoming_map, declared_keys = build_adjacency_structures(
                self.graph_definition,
                exclude_node_ids={"observer"},
            )
            # Convert to defaultdicts for backward compatibility
            self.adjacency_list = defaultdict(list, adj_list)
            self.incoming_edges_map = defaultdict(list, incoming_map)
            self.declared_output_keys = defaultdict(set, declared_keys)
        except GraphBuildError as e:
            log_status(f"[GraphOrchestrator] ERROR: {e}")
            raise ValueError(str(e)) from e

        # Use graph_builder module for topological ordering
        try:
            self.node_order = compute_topological_order(
                self.graph_definition,
                exclude_node_ids={"observer"},
            )
        except GraphBuildError as e:
            log_status(f"[GraphOrchestrator] ERROR: {e}")
            raise ValueError(str(e)) from e

        if not self.node_order:
            log_status("[GraphOrchestrator] WARNING: No nodes defined in the graph for sorting.")
        else:
            log_status(f"[GraphOrchestrator] INFO: Node execution order determined: {self.node_order}")

    def _initialize_agents(self):
        for node_def in self.graph_definition.get('nodes', []):
            agent_id = node_def['id']
            agent_type_name = node_def['type']
            agent_config_params = node_def.get('config', {})
            # A special type for providing initial inputs
            if agent_type_name == "InitialInputProvider":
                self.agents[agent_id] = None # No agent instance needed
                continue

            agent_class = get_agent_class(agent_type_name)
            if not agent_class:
                log_status(f"[GraphOrchestrator] ERROR: Unknown agent type: {agent_type_name} for node {agent_id}")
                raise ValueError(f"Unknown agent type: {agent_type_name} for node {agent_id}")
            try:
                self.agents[agent_id] = agent_class(agent_id, agent_type_name, agent_config_params, llm=self.llm, app_config=self.app_config)
            except Exception as e:  # pylint: disable=broad-exception-caught
                log_status(
                    f"[GraphOrchestrator] ERROR: Failed to initialize agent '{agent_id}' of type '{agent_type_name}': {e}")
                raise

    @staticmethod
    def _append_upstream_issue(agent_inputs: dict, target_key: str, source_node: str, message: str, fatal: bool) -> None:
        """Record upstream error details while preserving aggregated context."""

        if not message:
            return

        detail_entry = {
            "source": source_node,
            "target": target_key,
            "message": message,
        }
        agent_inputs.setdefault("upstream_error_details", []).append(detail_entry)

        messages_list = agent_inputs.setdefault("upstream_error_messages", [])
        if message not in messages_list:
            messages_list.append(message)

        if fatal:
            agent_inputs["error"] = "; ".join(messages_list)

    def _format_node_label(self, node: dict) -> Tuple[Optional[str], str]:
        """Return the node identifier and formatted label used in visualizations."""

        node_id = node.get("id")
        label = node.get("type", node_id)
        metrics = self.node_metrics.get(node_id)
        if metrics:
            label += f"\n{metrics['execution_time_sec']:.2f}s, {metrics['tokens_used']} tok"
        return node_id, label

    def _get_node_color(self, state: str, is_highlighted: bool) -> dict:
        """Return color configuration based on node state.

        Parameters
        ----------
        state : str
            One of 'pending', 'executing', 'completed', 'failed', 'skipped'.
        is_highlighted : bool
            Whether this node is currently highlighted.

        Returns
        -------
        dict
            Color configuration with 'background' and 'border' keys.
        """
        colors = {
            "pending": {"background": "#4A90D9", "border": "#2E6DA4"},
            "executing": {"background": "#FFB347", "border": "#FF8C00"},
            "completed": {"background": "#90EE90", "border": "#228B22"},
            "failed": {"background": "#FF6B6B", "border": "#CC0000"},
            "skipped": {"background": "#CCCCCC", "border": "#999999"},
        }

        if is_highlighted:
            return {"background": "#FFB347", "border": "#FF8C00"}

        return colors.get(state, colors["pending"])

    def _build_node_tooltip(self, node: dict) -> str:
        """Build rich HTML tooltip for a node.

        Parameters
        ----------
        node : dict
            Node definition from graph_definition.

        Returns
        -------
        str
            HTML string for the tooltip content.
        """
        node_id = node.get("id", "Unknown")
        node_type = node.get("type", "Unknown")
        config = node.get("config", {})
        description = config.get("description", "No description available")

        metrics = self.node_metrics.get(node_id, {})
        metrics_html = ""
        if metrics:
            exec_time = metrics.get("execution_time_sec", 0)
            tokens = metrics.get("tokens_used", 0)
            metrics_html = f"<hr/><b>Execution:</b> {exec_time:.2f}s<br/><b>Tokens:</b> {tokens}"

        # Get agent capabilities if available
        agent = self.agents.get(node_id)
        caps_html = ""
        if agent and hasattr(agent, "get_capabilities"):
            caps = agent.get_capabilities()
            if caps.needs_input_keys:
                caps_html += f"<br/><b>Inputs:</b> {', '.join(caps.needs_input_keys)}"
            if caps.requires_project_dir:
                caps_html += "<br/><i>Requires project directory</i>"

        return (
            f"<div style='padding:8px;max-width:300px;font-family:Arial,sans-serif;'>"
            f"<b style='font-size:14px;color:#333;'>{node_id}</b><br/>"
            f"<span style='color:#666;'>Type: {node_type}</span>"
            f"<hr style='margin:8px 0;border:none;border-top:1px solid #ddd;'/>"
            f"<div style='font-size:12px;color:#444;'>{description}</div>"
            f"{caps_html}{metrics_html}</div>"
        )

    def _inject_info_panel_script(self, html_file: str) -> None:
        """Inject JavaScript for click-to-show-info functionality.

        Parameters
        ----------
        html_file : str
            Path to the HTML file to modify.
        """
        info_panel_html = '''
<style>
    #info-panel {
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        height: 160px;
        background: linear-gradient(to bottom, #f8f9fa, #ffffff);
        border-top: 2px solid #4A90D9;
        padding: 15px 20px;
        overflow-y: auto;
        font-family: 'Segoe UI', Arial, sans-serif;
        box-shadow: 0 -4px 12px rgba(0,0,0,0.1);
        display: none;
        z-index: 1000;
    }
    #info-panel.visible { display: block; }
    #info-panel h3 {
        margin: 0 0 10px 0;
        color: #2E6DA4;
        font-size: 16px;
        border-bottom: 1px solid #e0e0e0;
        padding-bottom: 8px;
    }
    #info-content {
        font-size: 13px;
        color: #444;
        line-height: 1.5;
    }
    #close-panel {
        position: absolute;
        top: 10px;
        right: 15px;
        cursor: pointer;
        font-size: 18px;
        color: #999;
    }
    #close-panel:hover { color: #333; }
</style>
<div id="info-panel">
    <span id="close-panel" onclick="document.getElementById('info-panel').classList.remove('visible')">✕</span>
    <h3 id="info-title">Click a node to view agent details</h3>
    <div id="info-content"></div>
</div>
<script>
    network.on("click", function(params) {
        if (params.nodes.length > 0) {
            var nodeId = params.nodes[0];
            var node = network.body.data.nodes.get(nodeId);
            var panel = document.getElementById("info-panel");
            var title = document.getElementById("info-title");
            var content = document.getElementById("info-content");
            
            panel.classList.add("visible");
            title.innerHTML = "Agent: <strong>" + nodeId + "</strong>";
            content.innerHTML = node.title || "No details available for this node.";
        }
    });
    
    // Adjust network height to account for info panel
    document.getElementById("mynetwork").style.height = "calc(100vh - 180px)";
</script>
'''

        try:
            with open(html_file, "r", encoding="utf-8") as f:
                html_content = f.read()

            # Inject before closing body tag
            html_content = html_content.replace("</body>", info_panel_html + "</body>")

            with open(html_file, "w", encoding="utf-8") as f:
                f.write(html_content)
        except Exception as exc:  # pragma: no cover - best effort
            log_status(f"[GraphOrchestrator] WARNING: Could not inject info panel: {exc}")

    def _render_with_pyvis(
        self,
        output_path: str,
        highlight_node_id: Optional[str] = None,
        node_states: Optional[Dict[str, str]] = None,
    ) -> str:
        """Render the orchestrator graph using the pyvis network library.

        Parameters
        ----------
        output_path : str
            Base path (without extension) for the output file.
        highlight_node_id : Optional[str]
            Node ID to highlight (typically the executing node).
        node_states : Optional[Dict[str, str]]
            Map of node_id to state ('pending', 'executing', 'completed', 'failed', 'skipped').

        Returns
        -------
        str
            Path to the generated HTML file.
        """

        if Network is None:  # pragma: no cover - handled in ``visualize``
            raise RuntimeError("pyvis is not available")

        node_states = node_states or {}

        # Create network with enhanced settings
        network = Network(
            height="700px",
            width="100%",
            directed=True,
            bgcolor="#FAFAFA",
            font_color="#333333",
        )

        # Configure physics and layout for better visualization
        network.set_options('''
        {
            "nodes": {
                "font": {"size": 14, "face": "Segoe UI, Arial, sans-serif", "color": "#333"},
                "shadow": {"enabled": true, "size": 8, "x": 2, "y": 2}
            },
            "edges": {
                "arrows": {"to": {"enabled": true, "scaleFactor": 0.6}},
                "smooth": {"type": "cubicBezier", "forceDirection": "vertical"},
                "width": 1.5,
                "color": {"color": "#888888", "highlight": "#4A90D9", "hover": "#4A90D9"}
            },
            "physics": {
                "enabled": true,
                "hierarchicalRepulsion": {
                    "centralGravity": 0.0,
                    "springLength": 120,
                    "springConstant": 0.02,
                    "nodeDistance": 150
                },
                "solver": "hierarchicalRepulsion"
            },
            "layout": {
                "hierarchical": {
                    "enabled": true,
                    "direction": "UD",
                    "sortMethod": "directed",
                    "levelSeparation": 120,
                    "nodeSpacing": 180
                }
            },
            "interaction": {
                "hover": true,
                "tooltipDelay": 100,
                "zoomView": true,
                "dragView": true
            }
        }
        ''')

        added_node = False
        for node in self.graph_definition.get("nodes", []):
            node_id, label = self._format_node_label(node)
            if not node_id:
                continue
            added_node = True

            state = node_states.get(node_id, "pending")
            is_highlighted = node_id == highlight_node_id
            color = self._get_node_color(state, is_highlighted)
            tooltip = self._build_node_tooltip(node)

            # Determine border width based on state
            border_width = 3 if is_highlighted else (2 if state in ("completed", "failed") else 1)

            network.add_node(
                node_id,
                label=label.replace("\n", "<br/>"),
                title=tooltip,
                color=color,
                shape="box",
                size=30,
                borderWidth=border_width,
                borderWidthSelected=4,
                font={"size": 12, "face": "Segoe UI, Arial"},
            )

        for edge in self.graph_definition.get("edges", []):
            from_node = edge.get("from")
            to_node = edge.get("to")
            if from_node and to_node:
                network.add_edge(from_node, to_node)

        if not added_node:
            log_status("[GraphOrchestrator] WARNING: No nodes defined to visualize.")

        output_file = output_path + ".html"
        directory = os.path.dirname(output_file)
        if directory:
            os.makedirs(directory, exist_ok=True)

        network.write_html(output_file, notebook=False)

        # Inject the info panel for click-to-view-details
        self._inject_info_panel_script(output_file)

        log_status(
            f"[GraphOrchestrator] INFO: Graph visualization saved to {output_file} (pyvis)."
        )
        _open_graph_file(output_file)
        return output_file

    def _emit_event(self, name: str, payload: Dict[str, Any]) -> None:
        """Emit an event through the configured event bus."""
        self.event_bus.emit(name, payload)

    def visualize(
        self,
        output_path: str,
        output_format: str = "html",
        highlight_node_id: Optional[str] = None,
    ) -> str:
        """Generate a visual representation of the agent graph.

        Parameters
        ----------
        output_path: str
            Path (without extension) where the visualization will be written.
        output_format: str
            Ignored for PyVis rendering; kept for API compatibility.
        highlight_node_id: str | None
            Optional node identifier to highlight in the visualization.

        Returns
        -------
        str
            The path to the generated visualization file, or an empty string when
            rendering fails.
        """
        if Network is None:
            log_status(
                "[GraphOrchestrator] ERROR: PyVis is required for graph rendering. "
                "Install it with 'pip install pyvis'."
            )
            return ""

        try:
            output_file = self._render_with_pyvis(output_path, highlight_node_id)
            if output_file:
                self._emit_event(
                    EVENT_GRAPH_VISUALIZED,
                    {
                        "path": output_file,
                        "highlight_node_id": highlight_node_id,
                    },
                )
            return output_file
        except Exception as exc:  # pylint: disable=broad-exception-caught
            log_status(
                "[GraphOrchestrator] ERROR: Failed to render graph with PyVis: "
                f"{exc}"
            )
            log_status(traceback.format_exc())
            return ""

    def run(self, initial_inputs: Dict[str, Any], project_base_output_dir: str):
        """Execute the graph starting from the supplied ``initial_inputs``.

        Parameters
        ----------
        initial_inputs:
            Data provided to the special ``initial_input_provider`` node.
        project_base_output_dir:
            Directory into which agents may write artefacts.

        Returns
        -------
        Dict[str, Any]
            A mapping of node identifiers to their output dictionaries.
        """
        # pylint: disable=too-many-locals, too-many-branches, too-many-statements, too-many-nested-blocks
        outputs_history = {}
        self.node_metrics = {}
        os.makedirs(project_base_output_dir, exist_ok=True)
        self._emit_event(
            EVENT_WORKFLOW_STARTED,
            {
                "node_order": list(self.node_order),
                "initial_input_keys": list(initial_inputs.keys()),
                "project_base_output_dir": project_base_output_dir,
            },
        )
        # Visualize the initial graph once we know execution can start
        if self.node_order:
            self.visualize(os.path.join(project_base_output_dir, "graph_initial"))
        log_status(f"[GraphOrchestrator] Starting workflow with initial inputs: {list(initial_inputs.keys())}")

        for node_id in self.node_order:
            if node_id == "initial_input_provider":
                outputs_history[node_id] = initial_inputs
                log_status(f"\n[GraphOrchestrator] POPULATED_INPUTS: '{node_id}' with initial data.")
                continue

            current_agent = self.agents.get(node_id)
            if current_agent is None:
                log_status(f"\n[GraphOrchestrator] SKIPPING_NODE: '{node_id}' (No agent instance).")
                self._emit_event(
                    EVENT_NODE_SKIPPED,
                    {"node_id": node_id, "reason": "no_agent_instance"},
                )
                continue

            log_status(f"\n[GraphOrchestrator] EXECUTING_NODE: '{node_id}' (Type: {current_agent.agent_type})")
            self._emit_event(
                EVENT_NODE_STARTED,
                {"node_id": node_id, "agent_type": current_agent.agent_type},
            )
            # Visualize with the current node highlighted
            self.visualize(
                os.path.join(project_base_output_dir, f"graph_{node_id}"),
                highlight_node_id=node_id,
            )

            # --- Input Gathering ---
            agent_inputs = {}
            for edge_def in self.incoming_edges_map.get(node_id, []):
                from_node_id = edge_def['from']
                source_outputs = outputs_history.get(from_node_id)

                if source_outputs is None:
                    log_status(
                        f"[GraphOrchestrator] INPUT_ERROR: Output from source node '{from_node_id}' not found for target '{node_id}'."
                    )
                    data_mapping = edge_def.get("data_mapping", {})
                    for _, target_key in data_mapping.items():
                        missing_msg = f"Input from '{from_node_id}' missing."
                        agent_inputs[target_key] = None
                        agent_inputs[f"{target_key}_error"] = True
                        agent_inputs[f"{target_key}_error_message"] = missing_msg
                        agent_inputs[f"{target_key}_source"] = from_node_id
                        self._append_upstream_issue(agent_inputs, target_key, from_node_id, missing_msg, fatal=True)
                    continue

                data_mapping = edge_def.get("data_mapping")
                if not data_mapping:
                    log_status(f"[GraphOrchestrator] WARNING: No data_mapping for edge from '{from_node_id}' to '{node_id}'. Merging all outputs.")
                    agent_inputs.update(source_outputs)
                else:
                    for src_key, target_key in data_mapping.items():
                        if src_key in source_outputs:
                            agent_inputs[target_key] = source_outputs[src_key]
                            agent_inputs[f"{target_key}_source"] = from_node_id
                            if source_outputs.get("error"):
                                agent_inputs[f"{target_key}_error"] = True
                                agent_inputs[f"{target_key}_error_message"] = source_outputs.get("error")
                                self._append_upstream_issue(
                                    agent_inputs,
                                    target_key,
                                    from_node_id,
                                    source_outputs.get("error"),
                                    fatal=False,
                                )
                        else:
                            is_optional_initial_input = False
                            if from_node_id == "initial_input_provider":
                                if src_key in self.optional_initial_inputs or target_key in self.optional_initial_inputs:
                                    is_optional_initial_input = True
                            if is_optional_initial_input:
                                log_status(
                                    f"[GraphOrchestrator] INFO: Optional initial input '{src_key}' not provided; continuing without it."
                                )
                                missing_key_msg = (
                                    f"Optional input '{src_key}' was not provided by 'initial_input_provider'."
                                )
                            else:
                                log_status(
                                    f"[GraphOrchestrator] INPUT_ERROR: Source key '{src_key}' not found in output of '{from_node_id}' for target '{node_id}'."
                                )
                                missing_key_msg = f"Key '{src_key}' missing from '{from_node_id}'."
                            agent_inputs[target_key] = None
                            agent_inputs[f"{target_key}_error"] = True
                            agent_inputs[f"{target_key}_error_message"] = missing_key_msg
                            agent_inputs[f"{target_key}_source"] = from_node_id
                            self._append_upstream_issue(
                                agent_inputs,
                                target_key,
                                from_node_id,
                                missing_key_msg,
                                fatal=not is_optional_initial_input,
                            )

            # Special case for experimental data loader to get path from initial inputs if not connected by an edge
            if isinstance(current_agent, ExperimentalDataLoaderAgent) and "experimental_data_file_path" not in agent_inputs:
                if initial_inputs.get("experimental_data_file_path"):
                    agent_inputs["experimental_data_file_path"] = initial_inputs["experimental_data_file_path"]

            # --- Memory Agent Path Injection ---
            # For agents that need to write to the project directory, inject the base path.
            if current_agent.agent_type in ["LongTermMemoryAgent", "ShortTermMemoryAgent", "PDFSummaryWriterAgent"]:
                agent_inputs["project_base_output_dir"] = project_base_output_dir
                log_status(f"[{node_id}] INFO: Injected 'project_base_output_dir' for persistent storage.")

            if isinstance(current_agent, KnowledgeIntegratorAgent):
                aggregated_outputs = {
                    prev_node: copy.deepcopy(output)
                    for prev_node, output in outputs_history.items()
                    if prev_node not in {node_id, "observer"} and output is not None
                }
                if aggregated_outputs:
                    agent_inputs["all_agent_outputs"] = aggregated_outputs
                    log_status(
                        f"[{node_id}] INFO: Injected 'all_agent_outputs' with {len(aggregated_outputs)} upstream entries."
                    )

            log_status(f"[{node_id}] INFO: Inputs gathered: {{ {', '.join([f'{k}: {str(v)[:60]}...' for k,v in agent_inputs.items()])} }}")

            # --- Pre-execution Check for Upstream Failures ---
            aggregated_errors = agent_inputs.get("upstream_error_messages", [])
            upstream_error_message = agent_inputs.get("error")
            allow_partial = current_agent.config_params.get("allow_execution_with_errors", False)
            if upstream_error_message and not allow_partial:
                combined_error = upstream_error_message
                if aggregated_errors:
                    combined_error = "; ".join(aggregated_errors)
                log_status(
                    f"[GraphOrchestrator] SKIPPING_NODE: '{node_id}' due to upstream failure. Reason: {combined_error}"
                )
                self._emit_event(
                    EVENT_NODE_SKIPPED,
                    {"node_id": node_id, "reason": combined_error},
                )
                default_output = {"error": f"Skipped due to upstream failure: {combined_error}"}
                for declared_key in self.declared_output_keys.get(node_id, []):
                    default_output.setdefault(declared_key, None)
                if aggregated_errors:
                    default_output["upstream_error_messages"] = list(aggregated_errors)
                if agent_inputs.get("upstream_error_details"):
                    default_output["upstream_error_details"] = list(agent_inputs["upstream_error_details"])
                outputs_history[node_id] = default_output
                continue

            policy = current_agent.config_params.get("failure_policy", self.failure_policy)
            retries = current_agent.config_params.get("retries", 0)
            max_attempts = retries + 1 if policy == "retry" else 1

            node_output = {}
            start_time = time.time()
            tokens_before = (
                self.llm.get_total_tokens_used()
                if hasattr(self.llm, "get_total_tokens_used")
                else 0
            )
            for attempt in range(1, max_attempts + 1):
                try:
                    loop_over_key = current_agent.config_params.get("loop_over")

                    if (
                        loop_over_key
                        and loop_over_key in initial_inputs
                        and loop_over_key not in agent_inputs
                    ):
                        agent_inputs[loop_over_key] = initial_inputs[loop_over_key]

                    if (
                        loop_over_key
                        and loop_over_key in agent_inputs
                        and isinstance(agent_inputs[loop_over_key], list)
                    ):
                        input_list = agent_inputs[loop_over_key]
                        loop_outputs = [None] * len(input_list)
                        log_status(
                            f"[{node_id}] LOOP_START: Iterating over {len(input_list)} items from '{loop_over_key}'."
                        )

                        item_input_key = current_agent.config_params.get("loop_item_input_key")

                        def build_iteration_inputs(
                            item,
                            base_inputs=agent_inputs,
                            key=item_input_key,
                            loop_key=loop_over_key,
                        ):
                            iteration_inputs = {
                                k: v for k, v in base_inputs.items() if k != loop_key
                            }
                            if key:
                                iteration_inputs[key] = item
                            elif isinstance(item, dict):
                                iteration_inputs.update(item)
                            else:
                                iteration_inputs["item"] = item
                            return iteration_inputs

                        if current_agent.config_params.get("parallel_execution"):
                            max_workers = current_agent.config_params.get("max_workers")
                            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                                futures = {
                                    executor.submit(
                                        current_agent.execute, build_iteration_inputs(item)
                                    ): idx
                                    for idx, item in enumerate(input_list)
                                }
                                for future in as_completed(futures):
                                    idx = futures[future]
                                    try:
                                        loop_outputs[idx] = future.result()
                                    except Exception as e:  # pylint: disable=broad-exception-caught
                                        loop_outputs[idx] = {
                                            "error": f"Parallel execution failed: {e}"
                                        }
                        else:
                            for i, item in enumerate(input_list):
                                log_status(f"[{node_id}] -> Loop {i+1}/{len(input_list)}")
                                loop_outputs[i] = current_agent.execute(
                                    build_iteration_inputs(item)
                                )
                        node_output = {"results": loop_outputs}
                    else:
                        node_output = current_agent.execute(agent_inputs)

                except Exception as agent_exec_e:  # pylint: disable=broad-exception-caught
                    detailed_traceback = traceback.format_exc()
                    log_status(
                        f"[GraphOrchestrator] NODE_EXECUTION_CRITICAL_ERROR: Node '{node_id}' failed during execute(): {agent_exec_e}\n{detailed_traceback}"
                    )
                    node_output = {
                        "error": f"Agent execution for '{node_id}' failed critically: {agent_exec_e}"
                    }

                if node_output.get("error") and policy == "retry" and attempt < max_attempts:
                    log_status(f"[{node_id}] RETRY_ATTEMPT_{attempt} failed; retrying...")
                    self._emit_event(
                        EVENT_NODE_RETRY,
                        {
                            "node_id": node_id,
                            "attempt": attempt,
                            "error": node_output.get("error"),
                        },
                    )
                    continue
                break

            elapsed = time.time() - start_time
            tokens_after = (
                self.llm.get_total_tokens_used()
                if hasattr(self.llm, "get_total_tokens_used")
                else 0
            )
            self.node_metrics[node_id] = {
                "execution_time_sec": elapsed,
                "tokens_used": tokens_after - tokens_before,
            }
            log_status(
                f"[GraphOrchestrator] NODE_METRICS: '{node_id}' time={elapsed:.2f}s tokens={tokens_after - tokens_before}"
            )
            outputs_history[node_id] = node_output
            self._emit_event(
                EVENT_NODE_COMPLETED,
                {
                    "node_id": node_id,
                    "elapsed_sec": elapsed,
                    "tokens_used": tokens_after - tokens_before,
                    "error": node_output.get("error"),
                },
            )
            if node_output.get("error"):
                self._emit_event(
                    EVENT_NODE_FAILED,
                    {"node_id": node_id, "error": node_output.get("error")},
                )
            log_status(f"[{node_id}] RESULT: {{ {', '.join([f'{k}: {str(v)[:70]}...' for k,v in node_output.items()])} }}")
            if node_output.get("error"):
                log_status(f"[GraphOrchestrator] NODE_EXECUTION_ERROR_REPORTED: Node '{node_id}': {node_output['error']}")
                if policy == "abort":
                    return outputs_history

        # --- Observer Agent (if configured) ---
        observer_agent = self.agents.get("observer")
        if observer_agent:
            log_status("\n[GraphOrchestrator] EXECUTING_NODE (Observer): 'observer'")
            observer_output = observer_agent.execute({"outputs_history": outputs_history})
            outputs_history["observer"] = observer_output
            if observer_output.get("errors_found"):
                log_status(f"[GraphOrchestrator] OBSERVER_DETECTED_ERRORS: {observer_output['errors']}")
        else:
            log_status("[GraphOrchestrator] INFO: 'observer' agent not configured; skipping global error review.")

        self._save_consolidated_outputs(outputs_history, project_base_output_dir)
        # Final visualization after workflow completion
        self.visualize(os.path.join(project_base_output_dir, "graph_final"))
        log_status("\n[GraphOrchestrator] INFO: INTEGRATED workflow execution completed.")
        self._emit_event(
            EVENT_WORKFLOW_COMPLETED,
            {
                "node_count": len(self.node_order),
                "outputs_keys": list(outputs_history.keys()),
            },
        )
        return outputs_history

    def _save_consolidated_outputs(self, outputs_history: dict, project_base_output_dir: str):
        """Save workflow outputs to configured directories.

        Uses the artifact_writer module to handle file persistence.
        """
        log_status(f"[GraphOrchestrator] Saving consolidated outputs to {project_base_output_dir}...")

        # Create artifact writer with system configuration
        system_vars = self.app_config.get("system_variables", {})
        writer = ArtifactWriter(project_base_output_dir, system_vars)

        # Delegate to the artifact_writer module
        write_consolidated_outputs(outputs_history, writer)


def run_project_orchestration(
    pdf_file_paths: list,
    experimental_data_path: str,
    project_base_output_dir: str,
    status_update_callback: callable,
    app_config: Dict[str, Any],
    user_query: Optional[str] = None,
):
    """High-level convenience wrapper that executes an entire project workflow."""
    set_status_callback(status_update_callback)

    if not app_config:
        final_error_msg = "Critical: app_config dictionary is not provided to run_project_orchestration."
        log_status(f"[MainWorkflow] ERROR: {final_error_msg}")
        return {"error": final_error_msg}

    resolved_user_query: Optional[str] = None
    if user_query is not None:
        if isinstance(user_query, str):
            user_query = user_query.strip()
        resolved_user_query = user_query or None
    if resolved_user_query is None:
        default_query = app_config.get("system_variables", {}).get("default_user_query")
        if isinstance(default_query, str):
            default_query = default_query.strip()
            if default_query:
                resolved_user_query = default_query
                log_status("[MainWorkflow] INFO: Using default user query from configuration.")

    if resolved_user_query:
        log_status("[MainWorkflow] INFO: User query provided for deep research stage.")

    # Record run configuration before execution
    run_id = generate_run_id()
    save_run(
        run_id,
        app_config,
        {
            "pdf_file_paths": pdf_file_paths,
            "experimental_data_path": experimental_data_path,
            "project_base_output_dir": project_base_output_dir,
            "user_query": resolved_user_query,
        },
    )
    log_status(f"[MainWorkflow] INFO: Assigned run_id '{run_id}' to this execution.")

    system_vars = app_config.get("system_variables", {})
    llm_client_type = system_vars.get("llm_client", "openai")

    if llm_client_type == "openai":
        openai_api_key_check = system_vars.get("openai_api_key")
        if not openai_api_key_check or openai_api_key_check in [
            "YOUR_OPENAI_API_KEY_NOT_IN_CONFIG",
            "YOUR_ACTUAL_OPENAI_API_KEY",
            "KEY",
            "YOUR_KEY",
        ]:
            api_key_error_msg = (
                "OpenAI API key missing or placeholder while 'openai' LLM client is selected. "
                "Provide a valid key or choose the 'fake' client for keyless testing."
            )
            log_status(f"[MainWorkflow] CONFIG_ERROR: {api_key_error_msg}")
            raise ValueError(api_key_error_msg)

    if not pdf_file_paths:
        log_status(
            "[MainWorkflow] INFO: No PDF files provided for processing. Workflow will proceed but may lack PDF-derived input.")
    else:
        for p_path in pdf_file_paths:
            if not os.path.exists(p_path):
                path_error_msg = f"Input PDF not found at '{p_path}'. Please check the path."
                log_status(f"[MainWorkflow] FILE_ERROR: {path_error_msg}")
                return {"error": path_error_msg}
    if not os.path.exists(project_base_output_dir):
        try:
            os.makedirs(project_base_output_dir, exist_ok=True)
            log_status(f"[MainWorkflow] Created project output directory: {project_base_output_dir}")
        except OSError as e:
            dir_error_msg = f"Could not create project output directory '{project_base_output_dir}': {e}"
            log_status(f"[MainWorkflow] DIRECTORY_ERROR: {dir_error_msg}")
            return {"error": dir_error_msg}
    llm = None
    try:
        # --- LLM Client Factory ---
        api_key = system_vars.get("openai_api_key")
        timeout = float(system_vars.get("openai_api_timeout_seconds", 120))

        log_status(f"[MainWorkflow] INFO: Attempting to initialize LLM client of type '{llm_client_type}'.")

        try:
            llm = create_llm_client(app_config)
            log_status(f"[MainWorkflow] INFO: Initialized {llm_client_type} LLM client.")
        except ValueError as llm_err:
            raise ValueError(f"Failed to create LLM client: {llm_err}") from llm_err
        # --- End LLM Client Factory ---

        event_bus = EventBus()
        event_bus.subscribe("*", EventLogger().handle)
        metrics_path = os.path.join(project_base_output_dir, "node_metrics.json")
        event_bus.subscribe("*", MetricsCollector(output_path=metrics_path).handle)

        orchestrator = GraphOrchestrator(
            app_config.get("graph_definition"),
            llm,
            app_config,
            event_bus=event_bus,
        )

        initial_inputs = {
            "all_pdf_paths": pdf_file_paths,
            "experimental_data_file_path": experimental_data_path,
        }
        if resolved_user_query:
            initial_inputs["user_query"] = resolved_user_query

        final_outputs = orchestrator.run(
            initial_inputs=initial_inputs,
            project_base_output_dir=project_base_output_dir
        )
        final_outputs["run_id"] = run_id
        log_status("[MainWorkflow] INTEGRATED project orchestration finished.")
        return final_outputs
    except ValueError as ve:
        log_status(f"[MainWorkflow] ORCHESTRATION_SETUP_ERROR: {ve}")
        return {"error": f"Orchestration setup failed: {ve}"}
    except Exception as e:  # pylint: disable=broad-exception-caught
        detailed_traceback = traceback.format_exc()
        log_status(f"[MainWorkflow] UNEXPECTED_ORCHESTRATION_ERROR: Orchestration failed: {e}\n{detailed_traceback}")
        return {"error": f"Unexpected error during orchestration: {e}"}
    finally:
        if llm and hasattr(llm, "close"):
            llm.close()
