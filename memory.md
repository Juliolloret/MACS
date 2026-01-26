# Agent Memory

## Introduction

Agents often need more context than what fits in a single prompt. Memory lets
an agent store information, search it later, and reuse it across tasks. MACS
implements this idea with two agents that build and manage vector stores.

## How MACS Uses Memory (Simple View)

MACS has two memory agents:

1. ShortTermMemoryAgent
   - Builds a FAISS vector store for the current run.
   - Uses summaries from the PDF pipeline as input.
   - Saves the store inside the project output folder.
   - Output: `vector_store_path` (used by DeepResearchSummarizerAgent).

2. LongTermMemoryAgent
   - Loads a persistent FAISS store from disk.
   - Appends new summaries to the store so knowledge grows over time.
   - Saves the updated store back to disk.

## Memory Data Flow (Example)

```
PDFLoaderAgent -> PDFSummarizerAgent -> ShortTermMemoryAgent
                                     -> DeepResearchSummarizerAgent
                                     -> LongTermMemoryAgent
```

- The short-term store supports the deep research query for the current run.
- The long-term store accumulates knowledge across runs.

## Key Config Settings

These settings live in `config.json` under `system_variables`:

- `short_term_memory_filename` - filename for the short-term FAISS index.
- `long_term_memory_filename` - filename for the long-term FAISS index.
- `vector_store_path_key` and `storage_filename_key` - keys used by the memory
  agents to find these filenames.

Example:

```json
{
  "system_variables": {
    "short_term_memory_filename": "faiss_index_stm",
    "long_term_memory_filename": "faiss_index_ltm"
  }
}
```

## Dependencies

Memory agents require extra libraries:

- `faiss-cpu` for FAISS indexes
- `langchain-community` for vector store wrappers
- `langchain-openai` for embedding helpers (if using OpenAI embeddings)

If these packages are missing, the agent returns an `error` key instead of
raising a crash. Install with:

```bash
python -m pip install -r requirements.txt
```

## Common Issues and Fixes

- Error: "FAISS not installed" or missing vector store module
  - Fix: `python -m pip install faiss-cpu langchain-community`

- Error: "No vector store path provided"
  - Fix: Ensure ShortTermMemoryAgent runs before DeepResearchSummarizerAgent and
    the edge mapping includes `vector_store_path`.

- LongTermMemoryAgent does not change
  - Check if `individual_summaries` is empty; the agent skips updates when it
    has no new content.

## Glossary

- Embedding: a list of numbers that represent the meaning of text.
- Vector store: a database optimized for searching embeddings by similarity.
