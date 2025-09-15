"""Agents that manage short- and long-term vector store memories."""

import os
from typing import Dict, Any

try:
    from langchain_community.vectorstores import FAISS  # pylint: disable=import-error
except ImportError:
    FAISS = None

from utils import log_status
from .base_agent import Agent
from .registry import register_agent


@register_agent("ShortTermMemoryAgent")
class ShortTermMemoryAgent(Agent):
    """
    An agent that acts as a short-term memory buffer, collecting, structuring,
    and embedding information from other agents for semantic search.
    """

    def __init__(self, agent_id, agent_type, config_params=None, llm=None, app_config=None):  # pylint: disable=too-many-arguments,too-many-positional-arguments
        super().__init__(agent_id, agent_type, config_params, llm, app_config)
        self.vector_store_path_key = self.config_params.get("vector_store_path_key")
        if not self.vector_store_path_key:
            log_status(
                f"[{self.agent_id}] CRITICAL_WARNING: 'vector_store_path_key' not configured. "
                "The vector store path may not be saved correctly."
            )

    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Collects a list of summaries, generates embeddings, and saves them to a FAISS vector store.

        Args:
            inputs (dict): A dictionary expected to contain 'individual_summaries', a list of summary strings,
                           and 'project_base_output_dir'.

        Returns:
            dict: A dictionary containing the path to the FAISS index under 'vector_store_path'.
        """
        log_status(f"[{self.agent_id}] INFO: Short-term memory agent is processing inputs: {inputs}")

        individual_summaries_list = inputs.get("individual_summaries")
        if not isinstance(individual_summaries_list, list) or not individual_summaries_list:
            log_status(
                f"[{self.agent_id}] INFO: Input 'individual_summaries' is missing or empty. No STM will be created."
            )
            return {
                "vector_store_path": None,
                "individual_summaries": [],
                "error": "No individual summaries were provided to build short-term memory.",
            }

        # Extract summary text from the list of dictionaries
        valid_summaries = [s.get("summary") for s in individual_summaries_list if isinstance(s, dict) and s.get("summary")]
        log_status(f"[{self.agent_id}] INFO: Found {len(valid_summaries)} valid summaries.")

        if not valid_summaries:
            log_status(
                f"[{self.agent_id}] INFO: No valid summary strings found in 'individual_summaries'. No STM will be created."
            )
            return {
                "vector_store_path": None,
                "individual_summaries": [],
                "error": "No valid summary strings were found to build short-term memory.",
            }

        if FAISS is None:
            error_msg = "langchain_community.vectorstores is required but not installed."
            log_status(f"[{self.agent_id}] ERROR: {error_msg}")
            return {"error": error_msg}

        try:
            log_status(f"[{self.agent_id}] INFO: Generating embeddings for {len(valid_summaries)} summaries.")
            embeddings = self.llm.get_embeddings_client()

            # Create a FAISS vector store from the summaries
            vector_store = FAISS.from_texts(texts=valid_summaries, embedding=embeddings)
            log_status(f"[{self.agent_id}] INFO: FAISS vector store created.")

            # Determine the save path for the vector store
            base_dir = inputs.get("project_base_output_dir", ".")
            vector_store_filename = self.app_config.get("system_variables", {}).get(self.vector_store_path_key)
            if not vector_store_filename:
                log_status(
                    f"[{self.agent_id}] WARNING: Could not resolve filename from key '{self.vector_store_path_key}'. "
                    f"Defaulting to 'faiss_index_stm'."
                )
                vector_store_filename = "faiss_index_stm"

            save_path = os.path.join(base_dir, vector_store_filename)
            log_status(f"[{self.agent_id}] INFO: Vector store save path: {save_path}")
            os.makedirs(os.path.dirname(save_path), exist_ok=True)

            # Save the FAISS index
            vector_store.save_local(save_path)
            log_status(f"[{self.agent_id}] INFO: Successfully saved FAISS vector store to '{save_path}'.")

            return {"vector_store_path": save_path, "individual_summaries": valid_summaries}

        except Exception as e:  # pylint: disable=broad-except
            error_msg = f"Failed to generate or save embeddings: {e}"
            log_status(f"[{self.agent_id}] ERROR: {error_msg}")
            return {"error": error_msg}


@register_agent("LongTermMemoryAgent")
class LongTermMemoryAgent(Agent):
    """
    An agent that manages a persistent long-term memory using a vector store,
    integrating new knowledge from workflow runs.
    """

    def __init__(self, agent_id, agent_type, config_params=None, llm=None, app_config=None):  # pylint: disable=too-many-arguments,too-many-positional-arguments
        super().__init__(agent_id, agent_type, config_params, llm, app_config)
        self.storage_filename_key = self.config_params.get("storage_filename_key")
        if not self.storage_filename_key:
            log_status(
                f"[{self.agent_id}] CRITICAL_WARNING: 'storage_filename_key' not configured. "
                f"Long-term memory may not be persisted correctly."
            )

    def _get_storage_path(self, inputs: Dict[str, Any]) -> str:
        """Constructs the full storage path for the LTM vector store."""
        base_dir = inputs.get("project_base_output_dir", ".")
        filename = self.app_config.get("system_variables", {}).get(self.storage_filename_key)
        if not filename:
            log_status(
                f"[{self.agent_id}] WARNING: Could not resolve LTM filename from key '{self.storage_filename_key}'. "
                f"Defaulting to 'faiss_index_ltm'."
            )
            filename = "faiss_index_ltm"
        storage_path = os.path.join(base_dir, filename)
        log_status(f"[{self.agent_id}] INFO: LTM storage path: {storage_path}")
        return storage_path

    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Loads the long-term memory vector store, adds new summaries, and saves it back.

        Args:
            inputs (dict): A dictionary expected to contain 'individual_summaries' and 'project_base_output_dir'.

        Returns:
            dict: A dictionary containing the path to the updated LTM vector store.
        """
        log_status(f"[{self.agent_id}] INFO: Long-term memory agent is processing inputs: {inputs}")

        storage_path = self._get_storage_path(inputs)
        individual_summaries = inputs.get("individual_summaries")

        if not isinstance(individual_summaries, list) or not individual_summaries:
            log_status(f"[{self.agent_id}] INFO: No new summaries provided to add to long-term memory.")
            return {"long_term_memory_path": storage_path}

        valid_summaries = [s for s in individual_summaries if isinstance(s, str) and s.strip()]
        if not valid_summaries:
            log_status(f"[{self.agent_id}] INFO: No valid summary strings found to add to long-term memory.")
            return {"long_term_memory_path": storage_path}

        if FAISS is None:
            error_msg = "langchain_community.vectorstores is required but not installed."
            log_status(f"[{self.agent_id}] ERROR: {error_msg}")
            return {"error": error_msg}

        try:
            embeddings = self.llm.get_embeddings_client()
            vector_store = None

            # Load existing LTM vector store if it exists
            if os.path.exists(storage_path):
                log_status(f"[{self.agent_id}] INFO: Loading existing long-term memory from '{storage_path}'.")
                vector_store = FAISS.load_local(storage_path, embeddings, allow_dangerous_deserialization=True)
                log_status(f"[{self.agent_id}] INFO: Existing LTM loaded.")
                # Add new summaries to the vector store
                log_status(f"[{self.agent_id}] INFO: Adding {len(valid_summaries)} new summaries to long-term memory.")
                vector_store.add_texts(texts=valid_summaries)
                log_status(f"[{self.agent_id}] INFO: New summaries added to LTM.")
            else:
                log_status(f"[{self.agent_id}] INFO: No existing long-term memory found. Creating a new one from {len(valid_summaries)} summaries.")
                # Create a new store with the new summaries
                vector_store = FAISS.from_texts(texts=valid_summaries, embedding=embeddings)
                log_status(f"[{self.agent_id}] INFO: New LTM created.")

            # Save the updated vector store
            os.makedirs(os.path.dirname(storage_path), exist_ok=True)
            vector_store.save_local(storage_path)
            log_status(f"[{self.agent_id}] INFO: Successfully updated and saved long-term memory to '{storage_path}'.")

            return {"long_term_memory_path": storage_path}

        except Exception as e:  # pylint: disable=broad-except
            error_msg = f"Failed to update or save long-term memory: {e}"
            log_status(f"[{self.agent_id}] ERROR: {error_msg}")
            return {"error": error_msg}
