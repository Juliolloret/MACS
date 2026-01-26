"""Pluggable artifact writing utilities for workflow outputs."""

from __future__ import annotations

import os
from typing import Any, Callable, Dict, List, Optional

from utils import log_status


class ArtifactWriter:
    """Write workflow artifacts to configured output directories.

    This class provides a clean interface for persisting workflow outputs
    to disk, extracting the file writing logic from the orchestrator.
    """

    def __init__(
        self,
        base_output_dir: str,
        system_variables: Dict[str, Any],
    ):
        """Initialize the artifact writer.

        Parameters
        ----------
        base_output_dir : str
            Base directory for all output files.
        system_variables : Dict[str, Any]
            System configuration containing output folder names.
        """
        self.base_output_dir = base_output_dir
        self.output_paths = self._setup_output_directories(system_variables)

    def _setup_output_directories(
        self, system_variables: Dict[str, Any]
    ) -> Dict[str, str]:
        """Create and return mapping of output folder keys to paths.

        Parameters
        ----------
        system_variables : Dict[str, Any]
            Config dictionary containing output_project_*_folder_name keys.

        Returns
        -------
        Dict[str, str]
            Mapping from folder keys (e.g., 'synthesis') to full paths.
        """
        # Ensure base directory exists
        if not os.path.exists(self.base_output_dir):
            try:
                os.makedirs(self.base_output_dir, exist_ok=True)
            except OSError as e:
                log_status(
                    f"[ArtifactWriter] ERROR: Could not create base dir: {e}"
                )
                return {}

        output_paths = {}
        for key, subfolder_name in system_variables.items():
            if key.startswith("output_project_") and key.endswith("_folder_name"):
                if not isinstance(subfolder_name, str):
                    log_status(
                        f"[ArtifactWriter] WARNING: Invalid subfolder name for {key}"
                    )
                    continue

                # Extract folder key (e.g., 'synthesis' from 'output_project_synthesis_folder_name')
                folder_key = key.replace("output_project_", "").replace("_folder_name", "")
                full_path = os.path.join(self.base_output_dir, subfolder_name)
                output_paths[folder_key] = full_path

                if not os.path.exists(full_path):
                    try:
                        os.makedirs(full_path, exist_ok=True)
                    except OSError as e:
                        log_status(
                            f"[ArtifactWriter] ERROR: Could not create {full_path}: {e}"
                        )

        return output_paths

    def write_file(
        self,
        folder_key: str,
        filename: str,
        content: Optional[str],
        error: Optional[str] = None,
    ) -> bool:
        """Write content to a file in the specified output folder.

        Parameters
        ----------
        folder_key : str
            Key identifying the output folder (e.g., 'synthesis', 'hypotheses').
        filename : str
            Name of the file to create.
        content : Optional[str]
            Content to write. If None or empty, writes error or placeholder.
        error : Optional[str]
            Error message to write if content is empty.

        Returns
        -------
        bool
            True if file was written successfully, False otherwise.
        """
        folder_path = self.output_paths.get(folder_key)
        if not folder_path or not os.path.exists(folder_path):
            log_status(
                f"[ArtifactWriter] WARNING: Folder '{folder_key}' not available. "
                f"Cannot save '{filename}'."
            )
            return False

        file_path = os.path.join(folder_path, filename)
        try:
            if content in [None, ""]:
                data_to_write = f"Error: {error}" if error else "No content generated."
            else:
                data_to_write = str(content)

            with open(file_path, "w", encoding="utf-8") as f:
                f.write(data_to_write)

            placeholder_note = " (placeholder)" if content in [None, ""] and not error else ""
            log_status(f"[ArtifactWriter] Saved '{filename}' to '{folder_path}'{placeholder_note}")
            return True

        except OSError as e:
            log_status(f"[ArtifactWriter] ERROR writing {file_path}: {e}")
            return False

    def get_folder_path(self, folder_key: str) -> Optional[str]:
        """Get the full path for an output folder.

        Parameters
        ----------
        folder_key : str
            Key identifying the output folder.

        Returns
        -------
        Optional[str]
            Full path to the folder, or None if not configured.
        """
        return self.output_paths.get(folder_key)


def write_consolidated_outputs(
    outputs_history: Dict[str, Any],
    writer: ArtifactWriter,
) -> None:
    """Write all workflow outputs using the artifact writer.

    This function extracts the output writing logic that was previously
    in GraphOrchestrator._save_consolidated_outputs().

    Parameters
    ----------
    outputs_history : Dict[str, Any]
        Mapping of node IDs to their output dictionaries.
    writer : ArtifactWriter
        Configured artifact writer instance.
    """
    # Multi-document synthesis
    mds_out = outputs_history.get("multi_doc_synthesizer", {})
    writer.write_file(
        "synthesis",
        "multi_document_synthesis.txt",
        mds_out.get("multi_doc_synthesis_output"),
        mds_out.get("error"),
    )

    # Web research
    web_out = outputs_history.get("web_researcher", {})
    writer.write_file(
        "synthesis",
        "web_research_summary.txt",
        web_out.get("web_summary"),
        web_out.get("error"),
    )

    # Experimental data
    exp_out = outputs_history.get("experimental_data_loader", {})
    exp_data = exp_out.get("experimental_data_summary", "N/A")
    if exp_data != "N/A":
        writer.write_file(
            "synthesis",
            "experimental_data_summary.txt",
            exp_data,
            exp_out.get("error"),
        )

    # Knowledge brief
    kb_out = outputs_history.get("knowledge_integrator", {})
    writer.write_file(
        "synthesis",
        "integrated_knowledge_brief.txt",
        kb_out.get("integrated_knowledge_brief"),
        kb_out.get("error"),
    )

    # Hypotheses
    hypo_out = outputs_history.get("hypothesis_generator", {})
    writer.write_file(
        "hypotheses",
        "hypotheses_raw_llm_output.json",
        hypo_out.get("hypotheses_output_blob"),
        hypo_out.get("error"),
    )
    writer.write_file(
        "hypotheses",
        "key_research_opportunities.txt",
        hypo_out.get("key_opportunities"),
        hypo_out.get("error"),
    )

    # Hypotheses list
    hypo_list = hypo_out.get("hypotheses_list", [])
    if hypo_list:
        content = ""
        for i, item in enumerate(hypo_list):
            if isinstance(item, dict):
                content += f"{i + 1}. {item.get('hypothesis', 'N/A')}\n\n"
            else:
                content += f"{i + 1}. {item}\n\n"
        writer.write_file("hypotheses", "hypotheses_list.txt", content.strip())

    # Experiment designs
    _write_experiment_designs(outputs_history, writer)


def _write_experiment_designs(
    outputs_history: Dict[str, Any],
    writer: ArtifactWriter,
) -> None:
    """Write experiment design files.

    Parameters
    ----------
    outputs_history : Dict[str, Any]
        Mapping of node IDs to their output dictionaries.
    writer : ArtifactWriter
        Configured artifact writer instance.
    """
    exp_out = outputs_history.get("experiment_designer", {})
    designs_list = exp_out.get("experiment_designs_list", [])
    exp_path = writer.get_folder_path("experiments")

    if designs_list and exp_path and os.path.exists(exp_path):
        for i, design_info in enumerate(designs_list):
            hypo = design_info.get("hypothesis_processed", f"Hypothesis_N/A_{i + 1}")
            design = design_info.get("experiment_design", "")
            err = design_info.get("error")

            # Create safe filename
            safe_hypo = "".join(c if c.isalnum() else "_" for c in str(hypo)[:50]).strip("_")
            if not safe_hypo:
                safe_hypo = f"hypo_text_{i + 1}"
            filename = f"exp_design_{i + 1}_{safe_hypo}.txt"

            # Build content
            content = f"Hypothesis: {hypo}\n\n"
            if err:
                content += f"Error in design generation: {err}\n\n"
            content += f"Experiment Design:\n{design}\n"

            try:
                file_path = os.path.join(exp_path, filename)
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(content)
                log_status(f"[ArtifactWriter] Saved '{filename}' to experiments/")
            except OSError as e:
                log_status(f"[ArtifactWriter] ERROR writing {filename}: {e}")

    elif exp_out.get("error") and exp_path:
        writer.write_file(
            "experiments",
            "experiment_designs_error.txt",
            "",
            exp_out.get("error"),
        )
    else:
        log_status("[ArtifactWriter] INFO: No experiment designs to save.")
