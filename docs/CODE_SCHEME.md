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
  │ pdf_loader_agent.py → pdf_summarizer_agent.py → short_term_memory_agent.py │
  │             ├──→ deep_research_summarizer_agent.py → web_researcher_agent.py │
  │             └──→ long_term_memory_agent.py ─┐                               │
  │ experimental_data_loader_agent.py ──────────┴────────────┐                   │
  │                                   → knowledge_integrator_agent.py →        │
  │                                   → hypothesis_generator_agent.py →        │
  │                                   → experiment_designer_agent.py →         │
  │                                   → observer_agent.py                      │
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
5. **Agent pipeline** – PDFs are loaded and summarized, summaries are embedded in short-term memory, appended to long-term memory, queried for deep research synthesis, optionally enriched with web research and experimental data, integrated into a knowledge brief, used for hypothesis generation and experiment design, and finally reviewed by an observer agent.
6. **Frontends** – CLI (`cli_test.py`) or GUI (`gui.py`) launch the orchestrator with user-selected settings.
7. **Outputs** – each agent writes structured results to project directories for synthesis, hypotheses, experiments, etc.

This scheme provides a quick reference for understanding how MACS orchestrates its multi-agent workflow.
