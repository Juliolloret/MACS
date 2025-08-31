import os
from multi_agent_llm_system import GraphOrchestrator
from llm_fake import FakeLLM


def test_save_outputs_includes_error(tmp_path):
    app_config = {
        "system_variables": {
            "default_llm_model": "gpt-4o",
            "output_project_synthesis_folder_name": "synth",
            "output_project_hypotheses_folder_name": "hypo",
            "output_project_experiments_folder_name": "exp",
        }
    }
    orchestrator = GraphOrchestrator({"nodes": [], "edges": []}, FakeLLM(app_config), app_config)
    outputs_history = {
        "multi_doc_synthesizer": {
            "multi_doc_synthesis_output": "",
            "error": "synthesis failed",
        },
        "hypothesis_generator": {
            "hypotheses_output_blob": "",
            "key_opportunities": "",
            "error": "hypothesis failed",
        },
        "experiment_designer": {
            "experiment_designs_list": [],
            "error": "designer failed",
        },
    }
    orchestrator._save_consolidated_outputs(outputs_history, str(tmp_path))
    synth_file = tmp_path / "synth" / "multi_document_synthesis.txt"
    hypo_file = tmp_path / "hypo" / "hypotheses_raw_llm_output.json"
    exp_file = tmp_path / "exp" / "experiment_designs_error.txt"
    assert synth_file.read_text() == "Error: synthesis failed"
    assert hypo_file.read_text() == "Error: hypothesis failed"
    assert exp_file.read_text() == "Error: designer failed"
