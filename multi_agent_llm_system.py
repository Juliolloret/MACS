import os
import time
import json
from collections import defaultdict, deque
import asyncio
import traceback
from typing import List, Dict, Any, Optional

from llm import LLMClient

# --- SDK Imports & Availability Check ---
SDK_AVAILABLE = False
SDSAgent, Runner, WebSearchTool, ModelSettings = None, None, None, None
_sdk_import_error = None

try:
    from agents import (
        Agent as SDSAgent_actual,
        Runner as Runner_actual,
        WebSearchTool as WebSearchTool_actual,
        ModelSettings as ModelSettings_actual,
        set_default_openai_key  # Import the function to set the key for the SDK
    )

    SDSAgent, Runner, WebSearchTool, ModelSettings = (
        SDSAgent_actual, Runner_actual, WebSearchTool_actual, ModelSettings_actual
    )
    SDK_AVAILABLE = True
except ImportError as e:
    _sdk_import_error = e
    # Fallback if set_default_openai_key is not directly under agents in some versions
    if 'set_default_openai_key' not in globals():
        try:
            from agents.config import set_default_openai_key  # Common alternative path
        except ImportError:
            pass  # Will be handled by SDK_AVAILABLE check
    pass
# --- End SDK Imports & Availability Check ---


# --- End SDK Imports & Availability Check ---

# Import utilities from utils.py
from utils import (
    APP_CONFIG,
    REPORTLAB_AVAILABLE,  # Used in __main__
    load_app_config,
    log_status,
    set_status_callback,
)

# Imports for refactored Agent classes
# Core agent utilities
from agents import Agent, get_agent_class, ExperimentalDataLoaderAgent
# SDK Models are now in agents.sdk_models; WebResearcherAgent imports them directly.


# Script's Directory - this is specific to the main script file.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# SDK Availability Logging (using imported log_status)
if SDK_AVAILABLE:
    log_status("INFO: openai-agents SDK loaded successfully. SDK-based WebResearcherAgent capabilities enabled.")
else:
    error_message_suffix = f" (Error: {_sdk_import_error})" if _sdk_import_error else ""
    log_status(
        f"WARNING: openai-agents SDK not found or failed to import{error_message_suffix}. "
        "WebResearcherAgent with SDK integration will be disabled. "
        "Please ensure 'openai-agents' package is installed in the correct environment."
    )

# --- All class definitions (Agent, Pydantic models, specific agent classes) and utility functions are now removed from this file. ---
# They are imported from their respective locations in 'utils.py' or the 'agents' package.

class GraphOrchestrator:
    def __init__(self, graph_definition_from_config, llm: LLMClient):
        self.graph_definition = graph_definition_from_config
        self.agents = {}
        self.adjacency_list = defaultdict(list)
        self.node_order = []
        self.llm = llm
        self._build_graph_and_determine_order()
        self._initialize_agents()

    def _build_graph_and_determine_order(self):
        if 'nodes' not in self.graph_definition or 'edges' not in self.graph_definition:
            log_status("[GraphOrchestrator] ERROR: Graph definition in config must contain 'nodes' and 'edges'.")
            raise ValueError("Graph definition in config must contain 'nodes' and 'edges'.")
        node_ids = {node['id'] for node in self.graph_definition['nodes']}
        if not node_ids:
            log_status("[GraphOrchestrator] WARNING: No nodes defined in the graph.")
            self.node_order = []
            return
        in_degree = {node_id: 0 for node_id in node_ids}
        for edge in self.graph_definition.get('edges', []):
            from_node, to_node = edge.get('from'), edge.get('to')
            if from_node not in node_ids or to_node not in node_ids:
                log_status(f"[GraphOrchestrator] ERROR: Edge references undefined node: {from_node} -> {to_node}")
                raise ValueError(f"Edge references undefined node: {from_node} -> {to_node}")
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
            agent_class = get_agent_class(agent_type_name)
            if not agent_class:
                log_status(f"[GraphOrchestrator] ERROR: Unknown agent type: {agent_type_name} for node {agent_id}")
                raise ValueError(f"Unknown agent type: {agent_type_name} for node {agent_id}")
            try:
                self.agents[agent_id] = agent_class(agent_id, agent_type_name, agent_config_params, llm=self.llm)
            except Exception as e:
                log_status(
                    f"[GraphOrchestrator] ERROR: Failed to initialize agent '{agent_id}' of type '{agent_type_name}': {e}")
                raise

    def run(self, all_pdf_paths: list, experimental_data_file_path: str, project_base_output_dir: str):
        outputs_history = {}
        log_status(
            f"[GraphOrchestrator] Starting INTEGRATED workflow for {len(all_pdf_paths)} PDFs. Output dir: {project_base_output_dir}")

        # STAGE 1: PDF Processing
        all_summaries_for_synthesis = []
        pdf_loader_agent = self.agents.get("pdf_loader_node")
        pdf_summarizer_agent = self.agents.get("pdf_summarizer_node")
        if not pdf_loader_agent:
            log_status("[GraphOrchestrator] ERROR: 'pdf_loader_node' not found. Cannot process PDFs.")
            return {"error": "'pdf_loader_node' not defined or initialized."}
        if not pdf_summarizer_agent:
            log_status("[GraphOrchestrator] ERROR: 'pdf_summarizer_node' not found. Cannot summarize PDFs.")
            return {"error": "'pdf_summarizer_node' not defined or initialized."}

        for i, pdf_path in enumerate(all_pdf_paths):
            log_status(
                f"\n[GraphOrchestrator] Processing PDF {i + 1}/{len(all_pdf_paths)}: {os.path.basename(pdf_path)}")
            load_output = pdf_loader_agent.execute({"pdf_path": pdf_path})
            outputs_history[f"pdf_loader_{os.path.basename(pdf_path)}_{i}"] = load_output
            if load_output.get("error"):
                log_status(
                    f"[GraphOrchestrator] ERROR loading PDF {os.path.basename(pdf_path)}: {load_output['error']}")
                all_summaries_for_synthesis.append(
                    {"summary": f"Error loading {os.path.basename(pdf_path)}: {load_output['error']}",
                     "original_pdf_path": pdf_path, "error": True})
                continue
            summary_output = pdf_summarizer_agent.execute(
                {"pdf_text_content": load_output["pdf_text_content"], "original_pdf_path": pdf_path})
            outputs_history[f"pdf_summarizer_{os.path.basename(pdf_path)}_{i}"] = summary_output
            if summary_output.get("error"):
                log_status(
                    f"[GraphOrchestrator] ERROR summarizing PDF {os.path.basename(pdf_path)}: {summary_output['error']}")
                all_summaries_for_synthesis.append(
                    {"summary": f"Error summarizing {os.path.basename(pdf_path)}: {summary_output['error']}",
                     "original_pdf_path": pdf_path, "error": True})
            else:
                all_summaries_for_synthesis.append(
                    {"summary": summary_output["summary"], "original_pdf_path": pdf_path, "error": False})

        # STAGE 2: Multi-Document Synthesis (Explicit call before main graph loop)
        mds_output_data = {}
        if "multi_doc_synthesizer" in self.agents:
            log_status(f"\n[GraphOrchestrator] EXECUTING_NODE (Special Handling): 'multi_doc_synthesizer'")
            mds_agent = self.agents["multi_doc_synthesizer"]
            mds_inputs = {"all_pdf_summaries": all_summaries_for_synthesis}

            if not all_summaries_for_synthesis:
                log_status(
                    f"[{mds_agent.agent_id}] INPUT_ERROR: No PDF summaries collected for synthesis (list is empty).")
                mds_output_data = {"multi_doc_synthesis_output": "",
                                   "error": "No PDF summaries collected for synthesis."}
            else:
                mds_output_data = mds_agent.execute(mds_inputs)

            outputs_history["multi_doc_synthesizer"] = mds_output_data
            log_status(
                f"[multi_doc_synthesizer] RESULT: {{ {', '.join([f'{k}: {str(v)[:70]}...' for k, v in mds_output_data.items()])} }}")
            if mds_output_data.get("error"):
                log_status(
                    f"[GraphOrchestrator] NODE_EXECUTION_ERROR_REPORTED: multi_doc_synthesizer: {mds_output_data['error']}")
        else:
            log_status(
                "[GraphOrchestrator] WARNING: 'multi_doc_synthesizer' agent not found. Multi-document synthesis step will be skipped.")
            outputs_history["multi_doc_synthesizer"] = {"error": "MultiDocSynthesizerAgent not configured or found.",
                                                        "multi_doc_synthesis_output": ""}

        # STAGE 3: Execute remaining graph nodes based on topological order
        nodes_already_explicitly_handled = {"pdf_loader_node", "pdf_summarizer_node", "multi_doc_synthesizer", "observer"}

        for node_id in self.node_order:
            if node_id in nodes_already_explicitly_handled:
                continue
            if node_id not in self.agents:
                log_status(
                    f"[GraphOrchestrator] WARNING: Node '{node_id}' from order not found in initialized agents. Skipping.")
                outputs_history[node_id] = {"error": f"Agent for node '{node_id}' not initialized."}
                continue

            log_status(f"\n[GraphOrchestrator] EXECUTING_NODE: '{node_id}' (Type: {self.agents[node_id].agent_type})")
            current_agent = self.agents[node_id]
            agent_inputs = {}
            has_input_error = False
            for edge_def in self.graph_definition.get('edges', []):
                if edge_def['to'] == node_id:
                    from_node_id = edge_def['from']
                    source_outputs = outputs_history.get(from_node_id)
                    if not source_outputs:
                        log_status(
                            f"[GraphOrchestrator] INPUT_ERROR: Output from source node '{from_node_id}' not found for target '{node_id}'.")
                        data_mapping = edge_def.get("data_mapping", {})
                        for _, target_key in data_mapping.items():
                            agent_inputs[target_key] = f"Error: Input from '{from_node_id}' missing."
                            agent_inputs[f"{target_key}_error"] = True
                        has_input_error = True
                        continue
                    data_mapping = edge_def.get("data_mapping")
                    if not data_mapping:
                        log_status(
                            f"[GraphOrchestrator] WARNING: No data_mapping for edge from '{from_node_id}' to '{node_id}'. Merging all outputs.")
                        agent_inputs.update(source_outputs)
                        if source_outputs.get("error"): has_input_error = True
                    else:
                        for src_key, target_key in data_mapping.items():
                            if src_key in source_outputs:
                                agent_inputs[target_key] = source_outputs[src_key]
                                if source_outputs.get("error") or (
                                        isinstance(source_outputs[src_key], str) and source_outputs[src_key].startswith(
                                        "Error:")):
                                    agent_inputs[f"{target_key}_error"] = True
                                    has_input_error = True
                            else:
                                log_status(
                                    f"[GraphOrchestrator] INPUT_ERROR: Source key '{src_key}' not found in output of '{from_node_id}' for target '{node_id}'.")
                                agent_inputs[target_key] = f"Error: Key '{src_key}' missing from '{from_node_id}'."
                                agent_inputs[f"{target_key}_error"] = True
                                has_input_error = True
            if isinstance(current_agent,
                          ExperimentalDataLoaderAgent) and "experimental_data_file_path" not in agent_inputs:
                if experimental_data_file_path: agent_inputs[
                    "experimental_data_file_path"] = experimental_data_file_path
            log_status(
                f"[{node_id}] INFO: Inputs gathered: {{ {', '.join([f'{k}: {str(v)[:60]}...' for k, v in agent_inputs.items()])} }}")
            node_output = {}
            try:
                node_output = current_agent.execute(agent_inputs)
            except Exception as agent_exec_e:
                detailed_traceback = traceback.format_exc()
                log_status(
                    f"[GraphOrchestrator] NODE_EXECUTION_CRITICAL_ERROR: Node '{node_id}' failed during execute(): {agent_exec_e}\n{detailed_traceback}")
                node_output = {"error": f"Agent execution for '{node_id}' failed critically: {agent_exec_e}"}
            outputs_history[node_id] = node_output
            log_status(
                f"[{node_id}] RESULT: {{ {', '.join([f'{k}: {str(v)[:70]}...' for k, v in node_output.items()])} }}")
            if node_output.get("error"): log_status(
                f"[GraphOrchestrator] NODE_EXECUTION_ERROR_REPORTED: Node '{node_id}': {node_output['error']}")
        # Run observer agent at end if configured
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
        output_dirs_config = APP_CONFIG.get("system_variables", {})
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
            for i, h in enumerate(hypo_list): hypo_list_content += f"{i + 1}. {h}\n\n"
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
                              status_update_callback: callable, config_file_path: str = "config.json"):
    set_status_callback(status_update_callback)
    resolved_config_path = config_file_path
    if not os.path.isabs(config_file_path): resolved_config_path = os.path.join(SCRIPT_DIR, config_file_path)
    if not load_app_config(resolved_config_path):
        final_error_msg = f"Critical: Configuration load failed from '{resolved_config_path}'. Orchestration cannot proceed."
        log_status(f"[MainWorkflow] ERROR: {final_error_msg}")
        return {"error": final_error_msg}
    if not APP_CONFIG:
        final_error_msg = "Critical: APP_CONFIG is empty after load attempt. Orchestration cannot proceed."
        log_status(f"[MainWorkflow] ERROR: {final_error_msg}")
        return {"error": final_error_msg}

    # Set OpenAI API key for the SDK globally, if SDK is available
    if SDK_AVAILABLE and 'set_default_openai_key' in globals() and callable(set_default_openai_key):
        sdk_api_key = APP_CONFIG.get("system_variables", {}).get("openai_api_key")
        if sdk_api_key and sdk_api_key not in ["YOUR_OPENAI_API_KEY_NOT_IN_CONFIG", "YOUR_ACTUAL_OPENAI_API_KEY",
                                               "KEY"]:
            try:
                set_default_openai_key(sdk_api_key)
                log_status("[MainWorkflow] INFO: OpenAI API key set for openai-agents SDK.")
            except Exception as e:
                log_status(f"[MainWorkflow] WARNING: Failed to set OpenAI API key for SDK: {e}")
        else:
            log_status(
                "[MainWorkflow] WARNING: Valid OpenAI API key not found in APP_CONFIG to set for SDK. SDK calls might fail if OPENAI_API_KEY env var is not set.")
    elif SDK_AVAILABLE:
        log_status(
            "[MainWorkflow] WARNING: 'set_default_openai_key' function not available from SDK import. SDK calls might fail if OPENAI_API_KEY env var is not set.")

    openai_api_key_check = APP_CONFIG.get("system_variables", {}).get("openai_api_key")  # Renamed for clarity
    if not openai_api_key_check or openai_api_key_check in ["YOUR_OPENAI_API_KEY_NOT_IN_CONFIG",
                                                            "YOUR_ACTUAL_OPENAI_API_KEY", "KEY"]:
        api_key_error_msg = f"OpenAI API key missing or is a placeholder in configuration ('{resolved_config_path}'). Please set a valid API key for custom agent calls."
        log_status(
            f"[MainWorkflow] CONFIG_ERROR: {api_key_error_msg}")  # This is for your direct calls, SDK is separate

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
        from llm_openai import OpenAILLM

        api_key = APP_CONFIG.get("system_variables", {}).get("openai_api_key")
        timeout = float(APP_CONFIG.get("system_variables", {}).get("openai_api_timeout_seconds", 120))
        llm = OpenAILLM(api_key=api_key, timeout=int(timeout))
        orchestrator = GraphOrchestrator(APP_CONFIG.get("graph_definition"), llm)
        final_outputs = orchestrator.run(all_pdf_paths=pdf_file_paths,
                                         experimental_data_file_path=experimental_data_path,
                                         project_base_output_dir=project_base_output_dir)
        log_status("[MainWorkflow] INTEGRATED project orchestration finished.")
        return final_outputs
    except ValueError as ve:
        log_status(f"[MainWorkflow] ORCHESTRATION_SETUP_ERROR: {ve}")
        return {"error": f"Orchestration setup failed: {ve}"}
    except Exception as e:
        detailed_traceback = traceback.format_exc()
        log_status(f"[MainWorkflow] UNEXPECTED_ORCHESTRATION_ERROR: Orchestration failed: {e}\n{detailed_traceback}")
        return {"error": f"Unexpected error during orchestration: {e}"}


if __name__ == "__main__":
    CLI_CONFIG_FILENAME = "config_cli_test_integrated.json"
    cli_config_full_path = os.path.join(SCRIPT_DIR, CLI_CONFIG_FILENAME)
    if not os.path.exists(cli_config_full_path):
        print(f"[CLI_Test] '{CLI_CONFIG_FILENAME}' not found. Creating a minimal one for integrated test.")
        dummy_cli_config = {
            "system_variables": {"openai_api_key": "YOUR_OPENAI_API_KEY_IN_CLI_CONFIG",
                                 "output_project_synthesis_folder_name": "project_synthesis_cli",
                                 "output_project_hypotheses_folder_name": "project_hypotheses_cli",
                                 "output_project_experiments_folder_name": "project_experiments_cli",
                                 "default_llm_model": "gpt-3.5-turbo",
                                 "models": {"pdf_summarizer": "gpt-3.5-turbo",
                                            "multi_doc_synthesizer_model": "gpt-3.5-turbo",
                                            "web_research_model": "gpt-4o",
                                            "knowledge_integrator_model": "gpt-3.5-turbo",
                                            "hypothesis_generator": "gpt-3.5-turbo",
                                            "experiment_designer": "gpt-3.5-turbo",
                                            "sdk_planner_model": "gpt-3.5-turbo", "sdk_search_model": "gpt-3.5-turbo",
                                            "sdk_writer_model": "gpt-3.5-turbo"},
                                 "openai_api_timeout_seconds": 180},
            "agent_prompts": {"pdf_summarizer_sm": "Summarize this academic text in 1-2 paragraphs: {text_content}",
                              "multi_doc_synthesizer_sm": "Synthesize these summaries: {all_summaries_text}. Output: Cross-document understanding.",
                              "web_researcher_sm": "This agent uses an SDK for web research.",
                              "experimental_data_loader_sm": "This is experimental data: {data_content}. Present as a structured summary.",
                              "knowledge_integrator_sm": "Integrate: Multi-doc: {multi_doc_synthesis}, Web: {web_research_summary}, ExpData: {experimental_data_summary}. Output: Integrated brief.",
                              "hypothesis_generator_sm": "You are a highly insightful research strategist and innovator. Based on the provided 'integrated knowledge brief' (which consolidates information from multiple papers, web research, and experimental data), your task is to:\n1.  **Identify Key Opportunities:** Briefly highlight the most promising areas for novel research based on the integrated knowledge.\n2.  **Generate Hypotheses:** Propose {num_hypotheses} distinct, novel, and groundbreaking hypotheses that go beyond the current state-of-the-art. These hypotheses should be specific, testable, and aim to open new avenues of research by directly addressing the identified gaps or leveraging the novel connections found in the integrated brief.\n\n**Output Format STRICTLY REQUIRED:** Provide your response as a single JSON object with two keys:\n- `\"key_opportunities\"`: A string containing your brief summary of key research opportunities.\n- `\"hypotheses\"`: A JSON array of strings, where each string is a clearly articulated hypothesis.\n\nExample JSON output structure:\n```json\n{{\n  \"key_opportunities\": \"The integration of X from papers, Y from web, and Z from experiments points to a critical need to investigate C.\",\n  \"hypotheses\": [\n    \"Hypothesis 1: ...\",\n    \"Hypothesis 2: ...\"\n  ]\n}}\n```\nEnsure the entire output is a valid JSON object.",
                              "experiment_designer_sm": "Design experiment for: {hypothesis}",
                              "sdk_planner_sm": "Plan 2-3 web searches for query: {query}",
                              "sdk_searcher_sm": "Search web for: {search_term}. Reason: {reason}. Summarize results concisely.",
                              "sdk_writer_sm": "Write a brief report from query: {query} and search results: {search_results}."},
            "graph_definition": {"nodes": [
                {"id": "pdf_loader_node", "type": "PDFLoaderAgent", "config": {"description": "Loads text from PDF."}},
                {"id": "pdf_summarizer_node", "type": "PDFSummarizerAgent",
                 "config": {"model_key": "pdf_summarizer", "system_message_key": "pdf_summarizer_sm"}},
                {"id": "multi_doc_synthesizer", "type": "MultiDocSynthesizerAgent",
                 "config": {"model_key": "multi_doc_synthesizer_model",
                            "system_message_key": "multi_doc_synthesizer_sm"}},
                {"id": "web_researcher", "type": "WebResearcherAgent",
                 "config": {"model_key": "web_research_model", "system_message_key": "web_researcher_sm"}},
                {"id": "experimental_data_loader", "type": "ExperimentalDataLoaderAgent",
                 "config": {"system_message_key": "experimental_data_loader_sm"}},
                {"id": "knowledge_integrator", "type": "KnowledgeIntegratorAgent",
                 "config": {"model_key": "knowledge_integrator_model",
                            "system_message_key": "knowledge_integrator_sm"}},
                {"id": "hypothesis_generator", "type": "HypothesisGeneratorAgent",
                 "config": {"model_key": "hypothesis_generator", "system_message_key": "hypothesis_generator_sm",
                            "num_hypotheses": 3}},
                {"id": "experiment_designer", "type": "ExperimentDesignerAgent",
                 "config": {"model_key": "experiment_designer", "system_message_key": "experiment_designer_sm"}}],
                "edges": [
                    {"from": "multi_doc_synthesizer", "to": "web_researcher",
                     "data_mapping": {"multi_doc_synthesis_output": "cross_document_understanding"}},
                    {"from": "multi_doc_synthesizer", "to": "knowledge_integrator",
                     "data_mapping": {"multi_doc_synthesis_output": "multi_doc_synthesis"}},
                    {"from": "web_researcher", "to": "knowledge_integrator",
                     "data_mapping": {"web_summary": "web_research_summary"}},
                    {"from": "experimental_data_loader", "to": "knowledge_integrator",
                     "data_mapping": {"experimental_data_summary": "experimental_data_summary"}},
                    {"from": "knowledge_integrator", "to": "hypothesis_generator",
                     "data_mapping": {"integrated_knowledge_brief": "integrated_knowledge_brief"}},
                    {"from": "hypothesis_generator", "to": "experiment_designer",
                     "data_mapping": {"hypotheses_list": "hypotheses_list"}}
                ]}}
        try:
            with open(cli_config_full_path, 'w') as f:
                json.dump(dummy_cli_config, f, indent=2)
            print(
                f"[CLI_Test] Created '{cli_config_full_path}'. IMPORTANT: Please review and add a valid OpenAI API key.")
        except Exception as e:
            print(f"[CLI_Test] ERROR writing dummy CLI config: {e}")
            CLI_CONFIG_FILENAME = "config.json"
            cli_config_full_path = os.path.join(SCRIPT_DIR, CLI_CONFIG_FILENAME)
            print(f"[CLI_Test] Falling back to main '{CLI_CONFIG_FILENAME}'.")
    set_status_callback(print)
    if not load_app_config(config_path=cli_config_full_path):
        print(f"[CLI_Test] CRITICAL: Failed to load CLI config '{cli_config_full_path}'. Exiting.")
        exit(1)
    cli_api_key = APP_CONFIG.get("system_variables", {}).get("openai_api_key")
    if not cli_api_key or cli_api_key == "YOUR_OPENAI_API_KEY_IN_CLI_CONFIG": print(
        f"[CLI_Test] WARNING: OpenAI API key in '{cli_config_full_path}' is a placeholder. LLM calls will likely fail.")
    cli_test_input_dir = os.path.join(SCRIPT_DIR, "cli_test_multi_input_pdfs")
    cli_test_exp_data_dir = os.path.join(SCRIPT_DIR, "cli_test_experimental_data")
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    cli_test_project_output_dir = os.path.join(SCRIPT_DIR, f"cli_test_integrated_project_output_{timestamp}")
    for p_dir in [cli_test_input_dir, cli_test_exp_data_dir, cli_test_project_output_dir]:
        if not os.path.exists(p_dir):
            try:
                os.makedirs(p_dir, exist_ok=True)
            except OSError as e:
                print(f"[CLI_Test] ERROR creating directory {p_dir}: {e}. Exiting."); exit(1)
    pdf_paths_for_test = []
    if REPORTLAB_AVAILABLE:
        for i in range(2):
            pdf_name = f"dummy_paper_cli_{i + 1}.pdf"
            full_pdf_path = os.path.join(cli_test_input_dir, pdf_name)
            try:
                c = canvas.Canvas(full_pdf_path, pagesize=letter)
                text_obj = c.beginText(inch, 10 * inch);
                text_obj.setFont("Helvetica", 10)
                text_obj.textLine(f"CLI Test: Dummy PDF Document {i + 1}.");
                text_obj.textLine(f"This is page 1 of dummy document {i + 1}.")
                text_obj.textLine("It contains some sample text for testing PDF processing and summarization.")
                if i == 0:
                    text_obj.textLine(
                        "This document discusses advancements in AI-driven research methodologies."); text_obj.textLine(
                        "Keywords: artificial intelligence, machine learning, research automation, LLMs.")
                else:
                    text_obj.textLine(
                        "This document focuses on the challenges of multi-modal data integration."); text_obj.textLine(
                        "Keywords: data fusion, text analysis, image processing, knowledge graphs.")
                c.drawText(text_obj);
                c.showPage()
                text_obj = c.beginText(inch, 10 * inch);
                text_obj.setFont("Helvetica", 10)
                text_obj.textLine(f"Page 2 of dummy document {i + 1}.");
                text_obj.textLine("Further details and elaborations are usually found here.")
                c.drawText(text_obj);
                c.save()
                pdf_paths_for_test.append(full_pdf_path)
                print(f"[CLI_Test] Created dummy PDF: {full_pdf_path}")
            except Exception as e:
                print(f"[CLI_Test] ERROR creating dummy PDF {pdf_name}: {e}")
    else:
        print(
            "[CLI_Test] WARNING: reportlab library not found. Cannot create dummy PDFs for testing. Please create them manually in 'cli_test_multi_input_pdfs' or install reportlab.")
        for i in range(2):
            pdf_name = f"dummy_paper_cli_{i + 1}.pdf";
            full_pdf_path = os.path.join(cli_test_input_dir, pdf_name)
            if os.path.exists(full_pdf_path): pdf_paths_for_test.append(full_pdf_path); print(
                f"[CLI_Test] Using existing dummy PDF: {full_pdf_path}")
        if not pdf_paths_for_test: print(
            "[CLI_Test] No dummy PDFs found or created. PDF processing steps will likely fail.")
    exp_data_filename = "dummy_experimental_results_cli.txt"
    exp_data_full_path = os.path.join(cli_test_exp_data_dir, exp_data_filename)
    try:
        with open(exp_data_full_path, "w", encoding="utf-8") as f:
            f.write("Experimental Results Summary (CLI Test):\n");
            f.write("Experiment Alpha: Achieved 95% accuracy in classification task.\n")
            f.write("Experiment Beta: Showed a 20ms reduction in average latency.\n");
            f.write("Conclusion: The new model variant shows significant improvements.")
        print(f"[CLI_Test] Created dummy experimental data: {exp_data_full_path}")
    except Exception as e:
        print(f"[CLI_Test] ERROR creating dummy experimental data: {e}"); exp_data_full_path = None
    print(f"\n[CLI_Test] --- Running INTEGRATED project orchestration via CLI ---")
    print(
        f"[CLI_Test] Input PDFs ({len(pdf_paths_for_test)}): {cli_test_input_dir if pdf_paths_for_test else 'None (will use empty list)'}")
    print(f"[CLI_Test] Experimental data: {exp_data_full_path if exp_data_full_path else 'N/A'}")
    print(f"[CLI_Test] Project outputs will be in: {cli_test_project_output_dir}")
    print(f"[CLI_Test] Using config: {cli_config_full_path}\n")
    if not SDK_AVAILABLE: print(
        "[CLI_Test] WARNING: OpenAI Agents SDK is not installed. WebResearcherAgent will not function as intended.\nPlease run 'pip install openai-agents pydantic typing-extensions' to test full capabilities.")
    results = run_project_orchestration(pdf_file_paths=pdf_paths_for_test, experimental_data_path=exp_data_full_path,
                                        project_base_output_dir=cli_test_project_output_dir,
                                        status_update_callback=print, config_file_path=cli_config_full_path)
    print("\n" + "=" * 30 + " CLI INTEGRATED TEST FINAL RESULTS " + "=" * 30)
    if results and results.get("error"):
        print(f"CLI Test Run completed with an error: {results['error']}")
    elif results:
        print(f"CLI Test Run completed. Key outputs are in '{cli_test_project_output_dir}'.")
        final_ikb = results.get("knowledge_integrator", {}).get("integrated_knowledge_brief")
        if final_ikb:
            print(f"\nSnippet of Integrated Knowledge Brief:\n{str(final_ikb)[:300]}...")
        else:
            print("Integrated Knowledge Brief was not generated or found in results.")
    else:
        print("CLI Test Run did not produce a results dictionary (may have failed very early).")
    print("=" * (60 + len(" CLI INTEGRATED TEST FINAL RESULTS ")))
    print("\n--- Multi-Agent LLM System Backend CLI Integrated Test Finished ---")

