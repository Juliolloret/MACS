"""Agent for performing deep research summaries using a FAISS vector store."""

import os
from typing import Any, Dict

try:
    from langchain_community.vectorstores import FAISS
except ImportError:  # pragma: no cover
    FAISS = None  # type: ignore[assignment]

from utils import log_status
from .base_agent import Agent
from .registry import register_agent


@register_agent("DeepResearchSummarizerAgent")
class DeepResearchSummarizerAgent(Agent):
    """
    An agent that performs a deep research summary by querying a vector store
    and synthesizing the results with an LLM.
    """

    # pylint: disable=too-many-arguments, too-many-positional-arguments
    def __init__(self, agent_id, agent_type, config_params=None, llm=None, app_config=None):
        """Initialize the summarizer agent."""
        super().__init__(agent_id, agent_type, config_params, llm, app_config)
        self.top_k = self.config_params.get("top_k", 3)

    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Performs a semantic search on the vector store and synthesizes a response.

        Args:
            inputs (dict): A dictionary containing 'user_query' and 'vector_store_path'.

        Returns:
            dict: A dictionary containing the 'deep_research_summary'.
        """
        log_status(f"[{self.agent_id}] INFO: Deep research summarizer agent is processing inputs: {inputs}")

        user_query = inputs.get("user_query")
        if not user_query:
            return {"error": "Input 'user_query' was not provided."}

        vector_store_path = inputs.get("vector_store_path")
        if not vector_store_path:
            log_status(f"[{self.agent_id}] INFO: No vector store path provided. Skipping summarization.")
            return {"deep_research_summary": "No new documents were provided to generate a deep summary."}

        if not os.path.exists(vector_store_path):
            log_status(f"[{self.agent_id}] ERROR: Vector store not found at path: {vector_store_path}")
            return {"error": f"Vector store not found at path: {vector_store_path}"}

        if FAISS is None:
            error_msg = (
                "langchain_community.vectorstores is not available. Install the package to enable FAISS support."
            )
            log_status(f"[{self.agent_id}] ERROR: {error_msg}")
            return {"error": error_msg}

        try:
            log_status(f"[{self.agent_id}] INFO: Loading FAISS vector store from '{vector_store_path}'.")
            embeddings = self.llm.get_embeddings_client()
            vector_store = FAISS.load_local(vector_store_path, embeddings, allow_dangerous_deserialization=True)
            log_status(f"[{self.agent_id}] INFO: FAISS vector store loaded successfully.")

            log_status(f"[{self.agent_id}] INFO: Performing semantic search for query: '{user_query}'.")
            results = vector_store.similarity_search(user_query, k=self.top_k)
            log_status(f"[{self.agent_id}] INFO: Semantic search returned {len(results)} results.")

            if not results:
                return {"deep_research_summary": "No relevant information found for your query."}

            # Synthesize the results with an LLM
            prompt = self.get_formatted_system_message()
            context = "\n---\n".join([doc.page_content for doc in results])
            user_content = (
                f"Based on the following excerpts from multiple documents, please provide a comprehensive summary "
                f"that addresses the query: '{user_query}'.\n\nEXCERPTS:\n{context}"
            )

            log_status(f"[{self.agent_id}] INFO: Calling LLM to synthesize the deep research summary.")
            response = self.llm.get_response(
                system_message=prompt,
                user_message=user_content,
                model=self.model_name,
                temperature=self.config_params.get("temperature", 0.7),
            )
            log_status(f"[{self.agent_id}] INFO: LLM synthesis successful.")

            return {"deep_research_summary": response}

        except (OSError, ValueError, RuntimeError) as exc:
            error_msg = f"Failed during deep research summarization: {exc}"
            log_status(f"[{self.agent_id}] ERROR: {error_msg}")
            return {"error": error_msg}
