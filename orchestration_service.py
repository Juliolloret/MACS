"""High-level orchestration service for GUI and CLI integration.

This module provides a clean interface for running MACS workflows,
handling config loading, LLM setup, and output directory creation.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from graph_loader import load_config
from llm_factory import create_llm_client
from events import EventBus
from event_subscribers import EventLogger, MetricsCollector
from utils import log_status


@dataclass
class WorkflowConfig:
    """Configuration for a workflow run.

    Attributes
    ----------
    config_path : str
        Path to the config.json file.
    pdf_paths : List[str]
        List of PDF file paths to process.
    output_dir : str
        Base directory for output files.
    user_query : str
        The research query from the user.
    """

    config_path: str
    pdf_paths: List[str]
    output_dir: str
    user_query: str = ""
    project_name: Optional[str] = None


@dataclass
class WorkflowResult:
    """Result from running a workflow.

    Attributes
    ----------
    success : bool
        Whether the workflow completed successfully.
    output_dir : str
        Path to the output directory.
    outputs : Dict[str, Any]
        Outputs from each node.
    errors : List[str]
        Any errors encountered.
    metrics : Dict[str, Any]
        Execution metrics.
    """

    success: bool
    output_dir: str
    outputs: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)


class OrchestrationService:
    """High-level service for running MACS workflows.

    This service provides a clean separation between UI concerns
    (GUI, CLI) and the orchestration engine.

    Parameters
    ----------
    log_callback : Optional[Callable[[str], None]]
        Callback for log messages.
    """

    def __init__(
        self,
        log_callback: Optional[Callable[[str], None]] = None,
    ):
        self.log_callback = log_callback or log_status
        self._app_config: Optional[Dict[str, Any]] = None
        self._event_bus: Optional[EventBus] = None
        self._metrics_collector: Optional[MetricsCollector] = None

    def load_config(self, config_path: str) -> Dict[str, Any]:
        """Load and validate configuration.

        Parameters
        ----------
        config_path : str
            Path to config.json.

        Returns
        -------
        Dict[str, Any]
            The loaded configuration.
        """
        self._app_config = load_config_with_env_substitution(config_path)
        return self._app_config

    def run_workflow(self, config: WorkflowConfig) -> WorkflowResult:
        """Run a complete MACS workflow.

        Parameters
        ----------
        config : WorkflowConfig
            Configuration for the workflow.

        Returns
        -------
        WorkflowResult
            Result containing outputs, errors, and metrics.
        """
        # Import here to avoid circular imports
        from multi_agent_llm_system import run_project_orchestration

        # Load config if not already loaded
        if self._app_config is None:
            self.load_config(config.config_path)

        # Create output directory
        os.makedirs(config.output_dir, exist_ok=True)

        # Initialize event bus and subscribers
        self._event_bus = EventBus()
        event_logger = EventLogger(self.log_callback)
        self._event_bus.subscribe("*", event_logger.handle)

        metrics_path = os.path.join(config.output_dir, "node_metrics.json")
        self._metrics_collector = MetricsCollector(output_path=metrics_path)
        self._event_bus.subscribe("*", self._metrics_collector.handle)

        # Create LLM client
        llm = create_llm_client(self._app_config)

        # Run the workflow
        try:
            outputs = run_project_orchestration(
                pdf_file_paths=config.pdf_paths,
                user_query=config.user_query,
                app_config=self._app_config,
                project_base_output_dir=config.output_dir,
                llm=llm,
                event_bus=self._event_bus,
            )

            # Collect errors from outputs
            errors = []
            for node_id, output in outputs.items():
                if isinstance(output, dict) and output.get("error"):
                    errors.append(f"{node_id}: {output['error']}")

            return WorkflowResult(
                success=len(errors) == 0,
                output_dir=config.output_dir,
                outputs=outputs,
                errors=errors,
                metrics=self._metrics_collector.to_dict() if self._metrics_collector else {},
            )

        except Exception as e:
            return WorkflowResult(
                success=False,
                output_dir=config.output_dir,
                errors=[str(e)],
                metrics=self._metrics_collector.to_dict() if self._metrics_collector else {},
            )

    def get_metrics(self) -> Dict[str, Any]:
        """Get collected metrics from the last run.

        Returns
        -------
        Dict[str, Any]
            Metrics from the metrics collector.
        """
        if self._metrics_collector:
            return self._metrics_collector.to_dict()
        return {}

    def get_summary(self) -> Dict[str, object]:
        """Get summary of the last workflow run.

        Returns
        -------
        Dict[str, object]
            Summary including counts of nodes, errors, etc.
        """
        if self._metrics_collector:
            return self._metrics_collector.get_summary()
        return {}


def run_workflow_from_paths(
    config_path: str,
    pdf_paths: List[str],
    output_dir: str,
    user_query: str = "",
    log_callback: Optional[Callable[[str], None]] = None,
) -> WorkflowResult:
    """Convenience function to run a workflow.

    Parameters
    ----------
    config_path : str
        Path to config.json.
    pdf_paths : List[str]
        List of PDF file paths.
    output_dir : str
        Output directory.
    user_query : str
        Research query.
    log_callback : Optional[Callable[[str], None]]
        Log callback.

    Returns
    -------
    WorkflowResult
        Workflow result.
    """
    service = OrchestrationService(log_callback=log_callback)

    config = WorkflowConfig(
        config_path=config_path,
        pdf_paths=pdf_paths,
        output_dir=output_dir,
        user_query=user_query,
    )

    return service.run_workflow(config)
