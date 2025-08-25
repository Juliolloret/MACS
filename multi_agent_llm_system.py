import os
import json
from collections import defaultdict, deque
import traceback
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

from llm import LLMClient

# Import utilities from utils.py
from utils import (
    APP_CONFIG,
    load_app_config,
    log_status,
    set_status_callback,
)

# Configuration schema validation
from config_schema import validate_graph_definition

# Imports for refactored Agent classes
# Core agent utilities
from agents import Agent, get_agent_class
from agents.experimental_data_loader_agent import ExperimentalDataLoaderAgent
# SDK Models are now in agents.sdk_models; WebResearcherAgent imports them directly.


# Script's Directory - this is specific to the main script file.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# SDK availability is now checked and logged in utils.py

# --- All class definitions (Agent, Pydantic models, specific agent classes) and utility functions are now removed from this file. ---
# They are imported from their respective locations in 'utils.py' or the 'agents' package.

class GraphOrchestrator:
    def __init__(self, graph_definition_from_config, llm: LLMClient, app_config: Dict[str, Any]):
        # Validate and normalize the graph definition before proceeding
        self.graph_definition = validate_graph_definition(graph_definition_from_config)
        self.app_config = app_config
        self.agents = {}
        self.adjacency_list = defaultdict(list)
        self.incoming_edges_map = defaultdict(list)
        self.node_order = []
        self.llm = llm
        self._build_graph_and_determine_order()
        self._initialize_agents()

    def _build_graph_and_determine_order(self):
        if 'nodes' not in self.graph_definition or 'edges' not in self.graph_definition:
            log_status("[GraphOrchestrator] ERROR: Graph definition in config must contain 'nodes' and 'edges'.")
            raise ValueError("Graph definition in config must contain 'nodes' and 'edges'.")
        # Exclude the 'observer' node from the topological sort as it's handled as a special case after the main graph execution.
        graph_nodes = [n for n in self.graph_definition.get('nodes', []) if n.get('id') != 'observer']
        node_ids = {node['id'] for node in graph_nodes}

        if not node_ids:
            log_status("[GraphOrchestrator] WARNING: No nodes defined in the graph for sorting.")
            self.node_order = []
            return
        in_degree = {node_id: 0 for node_id in node_ids}
        all_node_ids_in_config = {node['id'] for node in self.graph_definition.get('nodes', [])} # For validation

        for edge in self.graph_definition.get('edges', []):
            from_node, to_node = edge.get('from'), edge.get('to')
            if from_node not in all_node_ids_in_config or to_node not in all_node_ids_in_config:
                log_status(f"[GraphOrchestrator] ERROR: Edge references undefined node ID: {from_node} -> {to_node}")
                raise ValueError(f"Edge references undefined node ID: {from_node} -> {to_node}")

            # Build lookup of incoming edges for each destination node
            self.incoming_edges_map[to_node].append(edge)

            if from_node in node_ids and to_node in node_ids:
                self.adjacency_list[from_node].append(to_node)
                in_degree[to_node] += 1
        queue = deque([node_id for node_id in node_ids if in_degree[node_id] == 0])
        self.node_order = []
        while queue:
            u = queue.popleft()
            self.node_order.append(u)
            for v_neighbor in self.adjacency_list.get(u, []):
                in_degree[v_neighbor] -= 1
                if in_degree[v_neighbor] == 0: queue.append(v_neighbor)
        if len(self.node_order) != len(node_ids):
            processed_nodes = set(self.node_order)
            missing_nodes = node_ids - processed_nodes
            log_status(
                f"[GraphOrchestrator] ERROR: Graph has a cycle or is disconnected. Order: {self.node_order}, Degrees: {in_degree}, Missing/Cyclic nodes: {missing_nodes}")
            raise ValueError(
                f"Graph has a cycle or is disconnected. Processed: {len(self.node_order)}/{len(node_ids)} nodes.")
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
            except Exception as e:
                log_status(
                    f"[GraphOrchestrator] ERROR: Failed to initialize agent '{agent_id}' of type '{agent_type_name}': {e}")
                raise

    def run(self, initial_inputs: Dict[str, Any], project_base_output_dir: str):
        outputs_history = {}
        log_status(f"[GraphOrchestrator] Starting workflow with initial inputs: {list(initial_inputs.keys())}")

        for node_id in self.node_order:
            if node_id == "initial_input_provider":
                outputs_history[node_id] = initial_inputs
                log_status(f"\n[GraphOrchestrator] POPULATED_INPUTS: '{node_id}' with initial data.")
                continue

            current_agent = self.agents.get(node_id)
            if current_agent is None:
                log_status(f"\n[GraphOrchestrator] SKIPPING_NODE: '{node_id}' (No agent instance).")
                continue

            log_status(f"\n[GraphOrchestrator] EXECUTING_NODE: '{node_id}' (Type: {current_agent.agent_type})")

            # --- Input Gathering ---
            agent_inputs = {}
            has_input_error = False
            for edge_def in self.incoming_edges_map.get(node_id, []):
                from_node_id = edge_def['from']
                source_outputs = outputs_history.get(from_node_id)

                if source_outputs is None:
                    log_status(f"[GraphOrchestrator] INPUT_ERROR: Output from source node '{from_node_id}' not found for target '{node_id}'.")
                    data_mapping = edge_def.get("data_mapping", {})
                    for _, target_key in data_mapping.items():
                        agent_inputs[target_key] = f"Error: Input from '{from_node_id}' missing."
                        agent_inputs[f"{target_key}_error"] = True
                    has_input_error = True
                    continue

                data_mapping = edge_def.get("data_mapping")
                if not data_mapping:
                    log_status(f"[GraphOrchestrator] WARNING: No data_mapping for edge from '{from_node_id}' to '{node_id}'. Merging all outputs.")
                    agent_inputs.update(source_outputs)
                    if source_outputs.get("error"): has_input_error = True
                else:
                    for src_key, target_key in data_mapping.items():
                        if src_key in source_outputs:
                            agent_inputs[target_key] = source_outputs[src_key]
                            if source_outputs.get("error") or (isinstance(source_outputs.get(src_key), str) and source_outputs[src_key].startswith("Error:")):
                                agent_inputs[f"{target_key}_error"] = True
                                has_input_error = True
                        else:
                            log_status(f"[GraphOrchestrator] INPUT_ERROR: Source key '{src_key}' not found in output of '{from_node_id}' for target '{node_id}'.")
                            agent_inputs[target_key] = f"Error: Key '{src_key}' missing from '{from_node_id}'."
                            agent_inputs[f"{target_key}_error"] = True
                            has_input_error = True

            # Special case for experimental data loader to get path from initial inputs if not connected by an edge
            if isinstance(current_agent, ExperimentalDataLoaderAgent) and "experimental_data_file_path" not in agent_inputs:
                if initial_inputs.get("experimental_data_file_path"):
                    agent_inputs["experimental_data_file_path"] = initial_inputs["experimental_data_file_path"]

            # --- Memory Agent Path Injection ---
            # For agents that need to write to the project directory, inject the base path.
            if current_agent.agent_type in ["LongTermMemoryAgent", "ShortTermMemoryAgent"]:
                agent_inputs["project_base_output_dir"] = project_base_output_dir
                log_status(f"[{node_id}] INFO: Injected 'project_base_output_dir' for persistent storage.")


            log_status(f"[{node_id}] INFO: Inputs gathered: {{ {', '.join([f'{k}: {str(v)[:60]}...' for k,v in agent_inputs.items()])} }}")

            # --- Execution ---
            node_output = {}
            try:
                # A more generic looping mechanism
                loop_over_key = current_agent.config_params.get("loop_over")

                # Check if the key for looping exists in the initial_inputs for root nodes
                if loop_over_key and loop_over_key in initial_inputs and not agent_inputs:
                    agent_inputs[loop_over_key] = initial_inputs[loop_over_key]

                if loop_over_key and loop_over_key in agent_inputs and isinstance(agent_inputs[loop_over_key], list):
                    input_list = agent_inputs[loop_over_key]
                    loop_outputs = [None] * len(input_list)
                    log_status(
                        f"[{node_id}] LOOP_START: Iterating over {len(input_list)} items from '{loop_over_key}'."
                    )

                    item_input_key = current_agent.config_params.get("loop_item_input_key")

                    def build_iteration_inputs(item):
                        iteration_inputs = {
                            k: v for k, v in agent_inputs.items() if k != loop_over_key
                        }
                        if item_input_key:
                            iteration_inputs[item_input_key] = item
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
                                except Exception as e:
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
                    # Standard execution for non-looping nodes
                    node_output = current_agent.execute(agent_inputs)

            except Exception as agent_exec_e:
                detailed_traceback = traceback.format_exc()
                log_status(f"[GraphOrchestrator] NODE_EXECUTION_CRITICAL_ERROR: Node '{node_id}' failed during execute(): {agent_exec_e}\n{detailed_traceback}")
                node_output = {"error": f"Agent execution for '{node_id}' failed critically: {agent_exec_e}"}

            outputs_history[node_id] = node_output
            log_status(f"[{node_id}] RESULT: {{ {', '.join([f'{k}: {str(v)[:70]}...' for k,v in node_output.items()])} }}")
            if node_output.get("error"):
                log_status(f"[GraphOrchestrator] NODE_EXECUTION_ERROR_REPORTED: Node '{node_id}': {node_output['error']}")

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
        log_status("\n[GraphOrchestrator] INFO: INTEGRATED workflow execution completed.")
        return outputs_history

    def _save_consolidated_outputs(self, outputs_history: dict, project_base_output_dir: str):
        log_status(f"[GraphOrchestrator] Saving consolidated outputs to {project_base_output_dir}...")
        output_dirs_config = self.app_config.get("system_variables", {})
        project_output_paths = {}
        if not os.path.exists(project_base_output_dir):
            try:
                os.makedirs(project_base_output_dir, exist_ok=True)
            except OSError as e:
                log_status(f"ERROR creating base project output directory {project_base_output_dir}: {e}")
                return
        for key, subfolder_name in output_dirs_config.items():
            if key.startswith("output_project_") and key.endswith("_folder_name"):
                if not isinstance(subfolder_name, str):
                    log_status(
                        f"[GraphOrchestrator] WARNING: Invalid subfolder name for {key}: {subfolder_name}. Skipping.")
                    continue
                full_path = os.path.join(project_base_output_dir, subfolder_name)
                project_output_paths[key.replace("output_project_", "").replace("_folder_name", "")] = full_path
                if not os.path.exists(full_path):
                    try:
                        os.makedirs(full_path, exist_ok=True)
                    except OSError as e:
                        log_status(f"ERROR creating output subfolder {full_path}: {e}")

        def write_output_file(folder_key, filename, content):
            folder_path = project_output_paths.get(folder_key)
            if content is not None and not (
                    isinstance(content, str) and content.startswith("Error:")) and folder_path and os.path.exists(
                    folder_path):
                try:
                    with open(os.path.join(folder_path, filename), "w", encoding="utf-8") as f:
                        f.write(str(content))
                    log_status(f"Saved '{filename}' to '{folder_path}'")
                except Exception as e:
                    log_status(f"ERROR writing file {os.path.join(folder_path, filename)}: {e}")
            elif not folder_path or not os.path.exists(folder_path):
                log_status(
                    f"WARNING: Output folder for key '{folder_key}' ('{folder_path}') does not exist. Cannot save '{filename}'.")
            elif content is None or (isinstance(content, str) and content.startswith("Error:")):
                log_status(
                    f"INFO: No valid content to save for '{filename}' (content was None, empty or an error string).")

        mds_out = outputs_history.get("multi_doc_synthesizer", {}).get("multi_doc_synthesis_output")
        write_output_file("synthesis", "multi_document_synthesis.txt", mds_out)
        web_research_out = outputs_history.get("web_researcher", {}).get("web_summary")
        write_output_file("synthesis", "web_research_summary.txt", web_research_out)
        exp_data_out = outputs_history.get("experimental_data_loader", {}).get("experimental_data_summary", "N/A")
        if exp_data_out != "N/A": write_output_file("synthesis", "experimental_data_summary.txt", exp_data_out)
        ikb_out = outputs_history.get("knowledge_integrator", {}).get("integrated_knowledge_brief")
        write_output_file("synthesis", "integrated_knowledge_brief.txt", ikb_out)
        hypo_gen_node_out = outputs_history.get("hypothesis_generator", {})
        raw_blob = hypo_gen_node_out.get("hypotheses_output_blob")
        write_output_file("hypotheses", "hypotheses_raw_llm_output.json", raw_blob)
        key_ops = hypo_gen_node_out.get("key_opportunities")
        write_output_file("hypotheses", "key_research_opportunities.txt", key_ops)
        hypo_list = hypo_gen_node_out.get("hypotheses_list", [])
        if hypo_list:
            hypo_list_content = ""
            # The structure of hypo_list can be a list of strings or dicts
            for i, h_item in enumerate(hypo_list):
                if isinstance(h_item, dict):
                    hypo_list_content += f"{i + 1}. {h_item.get('hypothesis', 'N/A')}\n\n"
                else:
                    hypo_list_content += f"{i + 1}. {h_item}\n\n"
            write_output_file("hypotheses", "hypotheses_list.txt", hypo_list_content.strip())
        exp_designs_list = outputs_history.get("experiment_designer", {}).get("experiment_designs_list", [])
        exp_path = project_output_paths.get("experiments")
        if exp_designs_list and exp_path and os.path.exists(exp_path):
            for i, design_info in enumerate(exp_designs_list):
                hypo = design_info.get("hypothesis_processed", f"Hypothesis_N/A_{i + 1}")
                design = design_info.get("experiment_design", "")
                err = design_info.get("error")
                safe_hypo_part = "".join(c if c.isalnum() else "_" for c in str(hypo)[:50]).strip('_')
                if not safe_hypo_part: safe_hypo_part = f"hypo_text_{i + 1}"
                fname = f"exp_design_{i + 1}_{safe_hypo_part}.txt"
                file_content = f"Hypothesis: {hypo}\n\n"
                if err: file_content += f"Error in design generation: {err}\n\n"
                file_content += f"Experiment Design:\n{design}\n"
                try:
                    with open(os.path.join(exp_path, fname), "w", encoding="utf-8") as f:
                        f.write(file_content)
                    log_status(f"Saved experiment design '{fname}' to '{exp_path}'")
                except Exception as e:
                    log_status(f"ERROR writing experiment design file {os.path.join(exp_path, fname)}: {e}")
        elif not exp_designs_list:
            log_status("INFO: No experiment designs were generated to save.")


def run_project_orchestration(pdf_file_paths: list, experimental_data_path: str, project_base_output_dir: str,
                              status_update_callback: callable, app_config: Dict[str, Any]):
    set_status_callback(status_update_callback)

    if not app_config:
        final_error_msg = "Critical: app_config dictionary is not provided to run_project_orchestration."
        log_status(f"[MainWorkflow] ERROR: {final_error_msg}")
        return {"error": final_error_msg}

    openai_api_key_check = app_config.get("system_variables", {}).get("openai_api_key")
    if not openai_api_key_check or openai_api_key_check in ["YOUR_OPENAI_API_KEY_NOT_IN_CONFIG",
                                                            "YOUR_ACTUAL_OPENAI_API_KEY", "KEY"]:
        api_key_error_msg = "OpenAI API key missing or is a placeholder in configuration. Please set a valid API key."
        log_status(f"[MainWorkflow] CONFIG_ERROR: {api_key_error_msg}")

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
    try:
        # --- LLM Client Factory ---
        system_vars = app_config.get("system_variables", {})
        llm_client_type = system_vars.get("llm_client", "openai")  # Default to 'openai'
        api_key = system_vars.get("openai_api_key")
        timeout = float(system_vars.get("openai_api_timeout_seconds", 120))
        llm = None

        log_status(f"[MainWorkflow] INFO: Attempting to initialize LLM client of type '{llm_client_type}'.")

        if llm_client_type == "openai":
            from llm_openai import OpenAILLM
            llm = OpenAILLM(app_config=app_config, api_key=api_key, timeout=int(timeout))
            log_status("[MainWorkflow] INFO: Initialized OpenAILLM client.")
        elif llm_client_type == "fake":
            from llm_fake import FakeLLM
            llm = FakeLLM(app_config=app_config)
            log_status("[MainWorkflow] INFO: Initialized FakeLLM client for testing.")
        else:
            raise ValueError(f"Unsupported LLM client type '{llm_client_type}' in configuration.")
        # --- End LLM Client Factory ---

        orchestrator = GraphOrchestrator(app_config.get("graph_definition"), llm, app_config)

        initial_inputs = {
            "all_pdf_paths": pdf_file_paths,
            "experimental_data_file_path": experimental_data_path,
        }

        final_outputs = orchestrator.run(
            initial_inputs=initial_inputs,
            project_base_output_dir=project_base_output_dir
        )
        log_status("[MainWorkflow] INTEGRATED project orchestration finished.")
        return final_outputs
    except ValueError as ve:
        log_status(f"[MainWorkflow] ORCHESTRATION_SETUP_ERROR: {ve}")
        return {"error": f"Orchestration setup failed: {ve}"}
    except Exception as e:
        detailed_traceback = traceback.format_exc()
        log_status(f"[MainWorkflow] UNEXPECTED_ORCHESTRATION_ERROR: Orchestration failed: {e}\n{detailed_traceback}")
        return {"error": f"Unexpected error during orchestration: {e}"}



