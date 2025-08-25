# MACS Codebase Scheme

This document provides an overview of how the MACS multi-agent research system operates.

## High-Level Flow

```
                     ┌───────────────────────────┐
                     │     CLI / GUI frontends   │
                     │   gui.py / cli_test.py    │
                     └─────────────┬─────────────┘
                                   │ (Project settings, PDF paths, etc.)
                                   ▼
                     ┌───────────────────────────┐
                     │  initial_input_provider   │
                     │ (Virtual node in graph)   │
                     └─────────────┬─────────────┘
                                   │
                                   ▼
┌────────────────────►┌───────────────────────────┐◄─────────────────────┐
│                     │     GraphOrchestrator     │                      │
│                     │ multi_agent_llm_system.py │                      │
│                     │ - Builds DAG from config  │                      │
│(Data flow between   │ - Executes agents         │   (Agent instances)  │
│      agents)        │ - Manages data I/O        │                      │
│                     └─────────────┬─────────────┘                      │
│                                   │ (Loads config)                     │
│                                   ▼                                    │
│                     ┌───────────────────────────┐                      │
│                     │        config.json        │                      │
│                     │ (Agent graph, prompts)    │                      │
│                     └───────────────────────────┘                      │
│                                                                        │
└───────────────────┐ ┌────────────────────────────────────────────────┐ │
                    │ │                  Agent Layer                   │ │
                    └─► │ (Dynamically loaded from `agents/` package)    │ ◄─┘
                        │                                                │
                        │ PDFLoader → PDFSummarizer → ShortTermMemory    │
                        │      │             └─────► DeepResearchSynth.  │
                        │      └───────────────────► LongTermMemory      │
                        │      └───────────────────► Exp. Data Loader    │
                        │                  └─────────────────────────┐   │
                        │                                            ▼   │
                        │                                KnowledgeIntegrator │
                        │                                            │   │
                        │                                            ▼   │
                        │                             HypothesisGenerator  │
                        │                                            │   │
                        │                                            ▼   │
                        │                               ExperimentDesigner │
                        └────────────────────────────────────────────────┘
```

## Execution Steps

1. **Frontend Interaction** – The user launches the workflow via the `gui.py` or `cli_test.py` script, providing settings like the input PDF directory and output path.
2. **Configuration Loading** – The orchestrator loads the `config.json` file, which defines the system prompts, model choices, and the all-important `graph_definition` (nodes and edges).
3. **Graph Building** – The `GraphOrchestrator` in `multi_agent_llm_system.py` parses the `graph_definition`. It identifies all agent nodes and the connections between them, building a directed acyclic graph (DAG). An `InitialInputProvider` virtual node is used to feed the user's settings into the graph.
4. **Agent Initialization** – For each node in the graph, the orchestrator dynamically loads the corresponding agent class from the `agents` package and creates an instance of it. A shared LLM client is passed to each agent.
5. **Topological Execution** – The orchestrator determines the correct execution order of the agents by performing a topological sort of the DAG.
6. **Agent Pipeline Execution** – The orchestrator executes the agents one by one, managing the data flow between them as defined by the edges in the graph. The typical pipeline is:
    - PDFs are loaded and summarized.
    - Summaries populate both a short-term (session) memory and a long-term (persistent) memory.
    - A deep, cross-document synthesis is performed using the short-term memory.
    - The synthesis, long-term memory, and optional experimental data are integrated into a final knowledge brief.
    - This brief is used to generate hypotheses and design experiments.
    - An observer agent reviews all outputs for errors.
7. **Output Generation** – Each agent saves its results (summaries, briefs, hypotheses, etc.) to structured subfolders in the designated project output directory.

This scheme provides a quick reference for understanding how MACS orchestrates its multi-agent workflow.
