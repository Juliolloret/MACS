from multi_agent_llm_system import GraphOrchestrator
from llm_fake import FakeLLM

def test_save_consolidated_outputs_writes_placeholders(tmp_path):
    app_config = {
        "system_variables": {
            "output_project_synthesis_folder_name": "project_synthesis",
            "output_project_hypotheses_folder_name": "project_hypotheses",
            "output_project_experiments_folder_name": "project_experiments",
        },
        "graph_definition": {"nodes": [], "edges": []},
    }
    orchestrator = GraphOrchestrator(app_config["graph_definition"], FakeLLM(app_config), app_config)
    outputs_history = {
        "multi_doc_synthesizer": {"multi_doc_synthesis_output": ""},
        "knowledge_integrator": {"integrated_knowledge_brief": ""},
        "hypothesis_generator": {
            "hypotheses_output_blob": "",
            "hypotheses_list": [],
            "key_opportunities": "",
        },
        "experiment_designer": {
            "experiment_designs_list": [
                {"hypothesis_processed": "Hypothesis1", "experiment_design": ""}
            ]
        },
    }
    orchestrator._save_consolidated_outputs(outputs_history, str(tmp_path))
    synthesis_file = tmp_path / "project_synthesis" / "multi_document_synthesis.txt"
    assert synthesis_file.read_text(encoding="utf-8") == "No content generated."
    hypotheses_file = tmp_path / "project_hypotheses" / "hypotheses_list.txt"
    assert hypotheses_file.read_text(encoding="utf-8") == "No content generated."
    exp_dir = tmp_path / "project_experiments"
    exp_files = list(exp_dir.iterdir())
    assert exp_files, "Experiment design file not created"
    exp_content = exp_files[0].read_text(encoding="utf-8")
    assert "No experiment design generated." in exp_content
