# MACS Codebase Scheme

This document provides an overview of how the MACS multi-agent research system operates.

## High-Level Flow

```
                       ┌────────────────────────┐
                       │      config.json       │
                       │  graph + model config  │
                       └──────────┬─────────────┘
                                  │ load_app_config
                                  ▼
                       ┌────────────────────────┐
                       │      LLM clients       │
                       │  llm.py / llm_openai   │
                       └──────────┬─────────────┘
                                  │ shared instance
                                  ▼
                       ┌────────────────────────┐
                       │       Agent base       │
                       │ agents/base_agent.py   │
                       │  - register decorator  │
                       │  - execute(model,...)  │
                       └──────────┬─────────────┘
                                  │ subclasses
                                  ▼
  ┌───────────────────────────────────────────────────────────────────────────┐
  │                        Agent implementations                              │
  │ pdf_loader_agent.py → pdf_summarizer_agent.py → multi_doc_synthesizer     │
  │        ↘ web_researcher_agent + experimental_data_loader_agent            │
  │              ↘ knowledge_integrator_agent → hypothesis_generator →        │
  │                   experiment_designer → observer_agent                    │
  └───────────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
                       ┌────────────────────────┐
                       │  GraphOrchestrator     │
                       │ multi_agent_llm_system │
                       │  - build DAG from config│
                       │  - topo order nodes     │
                       │  - manage data flows    │
                       └──────────┬─────────────┘
                                  │
                                  ▼
                       ┌────────────────────────┐
                       │    CLI / GUI frontends │
                       │ cli_test.py, gui.py    │
                       │  - parse flags         │
                       │  - run orchestrator    │
                       └──────────┬─────────────┘
                                  │
                                  ▼
                       ┌────────────────────────┐
                       │      Output folders    │
                       │ knowledge, synthesis…  │
                       └────────────────────────┘
```

## Execution Steps

1. **Configuration** – runtime options, model and prompt keys, and the workflow graph are loaded from `config.json`.
2. **LLM setup** – a single LLM client (real or fake) is initialized and shared across agents.
3. **Agent registration** – subclasses of `Agent` self-register via a decorator, allowing lookup by type name.
4. **Graph orchestration** – `GraphOrchestrator` builds a DAG from the config, resolves data edges, and executes nodes in topological order.
5. **Agent pipeline** – predefined agents implement the research workflow: PDFs are loaded and summarized, documents synthesized, optional web and experimental data integrated, hypotheses generated, experiments designed, and final results audited.
6. **Frontends** – CLI (`cli_test.py`) or GUI (`gui.py`) launch the orchestrator with user-selected settings.
7. **Outputs** – each agent writes structured results to project directories for synthesis, hypotheses, experiments, etc.

This scheme provides a quick reference for understanding how MACS orchestrates its multi-agent workflow.
