import os
import time
import json
from multi_agent_llm_system import run_project_orchestration, SCRIPT_DIR
from utils import REPORTLAB_AVAILABLE, set_status_callback, load_app_config

if REPORTLAB_AVAILABLE:
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.units import inch
    from reportlab.pdfgen import canvas
else:
    canvas = None
    letter = None
    inch = None

if __name__ == "__main__":
    CLI_CONFIG_FILENAME = "config_cli_test_integrated.json"
    cli_config_full_path = os.path.join(SCRIPT_DIR, CLI_CONFIG_FILENAME)

    if not os.path.exists(cli_config_full_path):
        print(f"[CLI_Test] '{CLI_CONFIG_FILENAME}' not found. Creating a minimal one for integrated test.")
        # Using the new graph definition structure
        dummy_cli_config = {
            "system_variables": {
                "openai_api_key": "YOUR_OPENAI_API_KEY_IN_CLI_CONFIG",
                "output_project_synthesis_folder_name": "project_synthesis_cli",
                "output_project_hypotheses_folder_name": "project_hypotheses_cli",
                "output_project_experiments_folder_name": "project_experiments_cli",
                "default_llm_model": "gpt-3.5-turbo",
                "models": {
                    "pdf_summarizer": "gpt-3.5-turbo",
                    "multi_doc_synthesizer_model": "gpt-3.5-turbo",
                    "web_research_model": "gpt-4o",
                    "knowledge_integrator_model": "gpt-3.5-turbo",
                    "hypothesis_generator": "gpt-3.5-turbo",
                    "experiment_designer": "gpt-3.5-turbo"
                },
                "openai_api_timeout_seconds": 180
            },
            "agent_prompts": {
                "pdf_summarizer_sm": "Summarize this academic text in 1-2 paragraphs: {text_content}",
                "multi_doc_synthesizer_sm": "Synthesize these summaries: {all_summaries_text}. Output: Cross-document understanding.",
                "web_researcher_sm": "This agent uses an SDK for web research.",
                "experimental_data_loader_sm": "This is experimental data: {data_content}. Present as a structured summary.",
                "knowledge_integrator_sm": "Integrate: Multi-doc: {multi_doc_synthesis}, Web: {web_research_summary}, ExpData: {experimental_data_summary}. Output: Integrated brief.",
                "hypothesis_generator_sm": "You are a highly insightful research strategist... (prompt content)",
                "experiment_designer_sm": "Design experiment for: {hypothesis}"
            },
            "graph_definition": {
                "nodes": [
                    {"id": "initial_input_provider", "type": "InitialInputProvider", "config": {}},
                    {"id": "pdf_loader_node", "type": "PDFLoaderAgent", "config": {"loop_over": "all_pdf_paths", "loop_item_input_key": "pdf_path"}},
                    {"id": "pdf_summarizer_node", "type": "PDFSummarizerAgent", "config": {"loop_over": "loader_results", "model_key": "pdf_summarizer", "system_message_key": "pdf_summarizer_sm"}},
                    {"id": "multi_doc_synthesizer", "type": "MultiDocSynthesizerAgent", "config": {"model_key": "multi_doc_synthesizer_model", "system_message_key": "multi_doc_synthesizer_sm"}},
                    {"id": "web_researcher", "type": "WebResearcherAgent", "config": {"model_key": "web_research_model", "system_message_key": "web_researcher_sm"}},
                    {"id": "experimental_data_loader", "type": "ExperimentalDataLoaderAgent", "config": {"system_message_key": "experimental_data_loader_sm"}},
                    {"id": "knowledge_integrator", "type": "KnowledgeIntegratorAgent", "config": {"model_key": "knowledge_integrator_model", "system_message_key": "knowledge_integrator_sm"}},
                    {"id": "hypothesis_generator", "type": "HypothesisGeneratorAgent", "config": {"model_key": "hypothesis_generator", "system_message_key": "hypothesis_generator_sm", "num_hypotheses": 3}},
                    {"id": "experiment_designer", "type": "ExperimentDesignerAgent", "config": {"model_key": "experiment_designer", "system_message_key": "experiment_designer_sm"}}
                ],
                "edges": [
                    {"from": "initial_input_provider", "to": "pdf_loader_node", "data_mapping": {"all_pdf_paths": "all_pdf_paths"}},
                    {"from": "pdf_loader_node", "to": "pdf_summarizer_node", "data_mapping": {"results": "loader_results"}},
                    {"from": "pdf_summarizer_node", "to": "multi_doc_synthesizer", "data_mapping": {"results": "all_pdf_summaries"}},
                    {"from": "multi_doc_synthesizer", "to": "web_researcher", "data_mapping": {"multi_doc_synthesis_output": "cross_document_understanding"}},
                    {"from": "multi_doc_synthesizer", "to": "knowledge_integrator", "data_mapping": {"multi_doc_synthesis_output": "multi_doc_synthesis"}},
                    {"from": "web_researcher", "to": "knowledge_integrator", "data_mapping": {"web_summary": "web_research_summary"}},
                    {"from": "experimental_data_loader", "to": "knowledge_integrator", "data_mapping": {"experimental_data_summary": "experimental_data_summary"}},
                    {"from": "knowledge_integrator", "to": "hypothesis_generator", "data_mapping": {"integrated_knowledge_brief": "integrated_knowledge_brief"}},
                    {"from": "hypothesis_generator", "to": "experiment_designer", "data_mapping": {"hypotheses_list": "hypotheses_list"}}
                ]
            }
        }
        try:
            with open(cli_config_full_path, 'w') as f:
                json.dump(dummy_cli_config, f, indent=2)
            print(f"[CLI_Test] Created '{cli_config_full_path}'. IMPORTANT: Please review and add a valid OpenAI API key.")
        except Exception as e:
            print(f"[CLI_Test] ERROR writing dummy CLI config: {e}")
            CLI_CONFIG_FILENAME = "config.json"
            cli_config_full_path = os.path.join(SCRIPT_DIR, CLI_CONFIG_FILENAME)
            print(f"[CLI_Test] Falling back to main '{CLI_CONFIG_FILENAME}'.")

    set_status_callback(print)
    app_config = load_app_config(config_path=cli_config_full_path)
    if not app_config:
        print(f"[CLI_Test] CRITICAL: Failed to load CLI config '{cli_config_full_path}'. Exiting.")
        exit(1)

    cli_api_key = app_config.get("system_variables", {}).get("openai_api_key")
    if not cli_api_key or cli_api_key == "YOUR_OPENAI_API_KEY_IN_CLI_CONFIG":
        print(f"[CLI_Test] WARNING: OpenAI API key in '{cli_config_full_path}' is a placeholder. LLM calls will likely fail.")

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
                text_obj = c.beginText(inch, 10 * inch)
                text_obj.setFont("Helvetica", 10)
                text_obj.textLine(f"CLI Test: Dummy PDF Document {i + 1}.")
                c.drawText(text_obj)
                c.save()
                pdf_paths_for_test.append(full_pdf_path)
                print(f"[CLI_Test] Created dummy PDF: {full_pdf_path}")
            except Exception as e:
                print(f"[CLI_Test] ERROR creating dummy PDF {pdf_name}: {e}")
    else:
        print("[CLI_Test] WARNING: reportlab library not found. Cannot create dummy PDFs for testing.")

    exp_data_filename = "dummy_experimental_results_cli.txt"
    exp_data_full_path = os.path.join(cli_test_exp_data_dir, exp_data_filename)
    try:
        with open(exp_data_full_path, "w", encoding="utf-8") as f:
            f.write("Experimental Results Summary (CLI Test).")
        print(f"[CLI_Test] Created dummy experimental data: {exp_data_full_path}")
    except Exception as e:
        print(f"[CLI_Test] ERROR creating dummy experimental data: {e}")
        exp_data_full_path = None

    print(f"\n[CLI_Test] --- Running INTEGRATED project orchestration via CLI ---")
    results = run_project_orchestration(
        pdf_file_paths=pdf_paths_for_test,
        experimental_data_path=exp_data_full_path,
        project_base_output_dir=cli_test_project_output_dir,
        status_update_callback=print,
        app_config=app_config
    )
    print("\n" + "=" * 30 + " CLI INTEGRATED TEST FINAL RESULTS " + "=" * 30)
    if results and results.get("error"):
        print(f"CLI Test Run completed with an error: {results['error']}")
    elif results:
        print(f"CLI Test Run completed. Key outputs are in '{cli_test_project_output_dir}'.")
    else:
        print("CLI Test Run did not produce a results dictionary.")
    print("=" * (60 + len(" CLI INTEGRATED TEST FINAL RESULTS ")))
    print("\n--- Multi-Agent LLM System Backend CLI Integrated Test Finished ---")
