"""Pydantic models for typed agent I/O contracts.

These models provide runtime validation and clear documentation of
expected inputs and outputs for each agent type.

Usage:
    from agents.io_models import PDFLoaderInput, PDFLoaderOutput
    
    # Validate input
    validated = PDFLoaderInput(**inputs)
    
    # Create typed output
    output = PDFLoaderOutput(pdf_text_content="...", original_pdf_path="...")
    return output.model_dump()
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# =============================================================================
# Base Models
# =============================================================================

class BaseAgentInput(BaseModel):
    """Base class for all agent inputs.
    
    Configured to allow extra fields for backward compatibility with
    existing workflows that may pass additional context.
    """
    
    class Config:
        extra = "allow"  # Allow unspecified fields for compatibility


class BaseAgentOutput(BaseModel):
    """Base class for all agent outputs.
    
    All outputs may optionally include an error field.
    """
    error: Optional[str] = Field(default=None, description="Error message if execution failed")
    
    class Config:
        extra = "allow"  # Allow unspecified fields for compatibility

    def has_error(self) -> bool:
        """Return True if an error was set."""
        return self.error is not None and len(self.error) > 0


# =============================================================================
# PDF Processing Contracts
# =============================================================================

class PDFLoaderInput(BaseAgentInput):
    """Input contract for PDFLoaderAgent."""
    pdf_path: str = Field(..., description="Path to the PDF file to load")


class PDFLoaderOutput(BaseAgentOutput):
    """Output contract for PDFLoaderAgent."""
    pdf_text_content: str = Field(default="", description="Extracted text content from PDF")
    original_pdf_path: Optional[str] = Field(default=None, description="Original path of the processed PDF")


class PDFSummarizerInput(BaseAgentInput):
    """Input contract for PDFSummarizerAgent."""
    pdf_text_content: str = Field(..., description="Text content extracted from a PDF")
    original_pdf_path: str = Field(default="Unknown PDF", description="Path to the original PDF")


class PDFSummarizerOutput(BaseAgentOutput):
    """Output contract for PDFSummarizerAgent."""
    summary: str = Field(default="", description="Generated summary of the PDF content")
    original_pdf_path: str = Field(default="", description="Path to the original PDF")


class SummaryWriterInput(BaseAgentInput):
    """Input contract for PDFSummaryWriterAgent."""
    summaries_to_write: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="List of summary dicts with 'summary' and 'original_pdf_path' keys"
    )
    project_base_output_dir: str = Field(..., description="Base directory for output files")


class SummaryWriterOutput(BaseAgentOutput):
    """Output contract for PDFSummaryWriterAgent."""
    written_summary_files: List[str] = Field(
        default_factory=list,
        description="Paths to written summary files"
    )


# =============================================================================
# Multi-Document Synthesis Contracts
# =============================================================================

class MultiDocSynthesizerInput(BaseAgentInput):
    """Input contract for MultiDocSynthesizerAgent."""
    all_pdf_summaries: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="List of summary objects from PDFSummarizerAgent"
    )


class MultiDocSynthesizerOutput(BaseAgentOutput):
    """Output contract for MultiDocSynthesizerAgent."""
    multi_doc_synthesis_output: str = Field(
        default="",
        description="Cross-document synthesis combining all summaries"
    )
    individual_summaries: List[str] = Field(
        default_factory=list,
        description="List of individual summary texts"
    )


# =============================================================================
# Memory Agent Contracts
# =============================================================================

class ShortTermMemoryInput(BaseAgentInput):
    """Input contract for ShortTermMemoryAgent."""
    individual_summaries: List[str] = Field(
        default_factory=list,
        description="List of summary texts to index"
    )
    project_base_output_dir: str = Field(..., description="Directory to save the vector store")


class ShortTermMemoryOutput(BaseAgentOutput):
    """Output contract for ShortTermMemoryAgent."""
    vector_store_path: str = Field(default="", description="Path to the created FAISS index")
    individual_summaries: List[str] = Field(
        default_factory=list,
        description="Pass-through of input summaries"
    )


class LongTermMemoryInput(BaseAgentInput):
    """Input contract for LongTermMemoryAgent."""
    individual_summaries: List[str] = Field(
        default_factory=list,
        description="New summaries to add to the store"
    )
    project_base_output_dir: str = Field(..., description="Directory for persistent storage")


class LongTermMemoryOutput(BaseAgentOutput):
    """Output contract for LongTermMemoryAgent."""
    long_term_memory_path: str = Field(default="", description="Path to the updated FAISS index")


# =============================================================================
# Research and Analysis Contracts
# =============================================================================

class DeepResearchSummarizerInput(BaseAgentInput):
    """Input contract for DeepResearchSummarizerAgent."""
    user_query: Optional[str] = Field(default=None, description="User's research question")
    vector_store_path: Optional[str] = Field(default=None, description="Path to FAISS index to search")


class DeepResearchSummarizerOutput(BaseAgentOutput):
    """Output contract for DeepResearchSummarizerAgent."""
    deep_research_summary: str = Field(default="", description="Research summary answering the query")
    skipped: bool = Field(default=False, description="Whether execution was skipped")


class WebResearcherInput(BaseAgentInput):
    """Input contract for WebResearcherAgent."""
    cross_document_understanding: Optional[str] = Field(
        default=None,
        description="Multi-document synthesis to expand upon"
    )


class WebResearcherOutput(BaseAgentOutput):
    """Output contract for WebResearcherAgent."""
    web_summary: str = Field(default="", description="Generated web research summary")


class ExperimentalDataLoaderInput(BaseAgentInput):
    """Input contract for ExperimentalDataLoaderAgent."""
    experimental_data_file_path: Optional[str] = Field(
        default=None,
        description="Path to experimental data file"
    )


class ExperimentalDataLoaderOutput(BaseAgentOutput):
    """Output contract for ExperimentalDataLoaderAgent."""
    experimental_data_summary: str = Field(
        default="",
        description="Summary or content of experimental data"
    )


# =============================================================================
# Knowledge Integration Contracts
# =============================================================================

class KnowledgeIntegratorInput(BaseAgentInput):
    """Input contract for KnowledgeIntegratorAgent."""
    multi_doc_synthesis: Optional[str] = Field(default=None, description="Multi-document synthesis")
    deep_research_summary: Optional[str] = Field(default=None, description="Deep research summary")
    web_research_summary: Optional[str] = Field(default=None, description="Web research summary")
    experimental_data_summary: Optional[str] = Field(default=None, description="Experimental data summary")
    all_agent_outputs: Optional[Dict[str, Any]] = Field(
        default=None,
        description="All upstream agent outputs for aggregation"
    )


class KnowledgeIntegratorOutput(BaseAgentOutput):
    """Output contract for KnowledgeIntegratorAgent."""
    integrated_knowledge_brief: str = Field(
        default="",
        description="Integrated knowledge brief combining all sources"
    )
    knowledge_sections: Dict[str, str] = Field(
        default_factory=dict,
        description="Individual knowledge sections by source"
    )
    contributing_agents: List[str] = Field(
        default_factory=list,
        description="List of agents that contributed content"
    )
    agent_context_details: Dict[str, str] = Field(
        default_factory=dict,
        description="Context details per agent"
    )


# =============================================================================
# Hypothesis and Experiment Design Contracts
# =============================================================================

class Hypothesis(BaseModel):
    """A single research hypothesis with justification."""
    hypothesis: str = Field(..., description="The hypothesis statement")
    justification: str = Field(..., description="Justification for the hypothesis")


class HypothesisGeneratorInput(BaseAgentInput):
    """Input contract for HypothesisGeneratorAgent."""
    integrated_knowledge_brief: str = Field(
        ...,
        description="Integrated knowledge brief to generate hypotheses from"
    )


class HypothesisGeneratorOutput(BaseAgentOutput):
    """Output contract for HypothesisGeneratorAgent."""
    hypotheses_output_blob: str = Field(
        default="",
        description="Raw LLM response containing hypotheses"
    )
    hypotheses_list: List[Dict[str, str]] = Field(
        default_factory=list,
        description="Parsed list of hypothesis objects"
    )
    key_opportunities: str = Field(
        default="",
        description="Key research opportunities identified"
    )


class ExperimentDesignerInput(BaseAgentInput):
    """Input contract for ExperimentDesignerAgent."""
    hypothesis: Optional[Dict[str, str]] = Field(
        default=None,
        description="Single hypothesis to design experiment for"
    )
    hypotheses_list: Optional[List[Dict[str, str]]] = Field(
        default=None,
        description="List of hypotheses to design experiments for"
    )


class ExperimentDesignerOutput(BaseAgentOutput):
    """Output contract for ExperimentDesignerAgent."""
    experiment_design: Optional[str] = Field(
        default=None,
        description="Single experiment design (when single hypothesis input)"
    )
    experiment_designs_list: List[str] = Field(
        default_factory=list,
        description="List of experiment designs (when multiple hypotheses)"
    )


# =============================================================================
# Observer Contract
# =============================================================================

class ObserverInput(BaseAgentInput):
    """Input contract for ObserverAgent."""
    outputs_history: Dict[str, Any] = Field(
        ...,
        description="Complete history of all node outputs"
    )


class ObserverOutput(BaseAgentOutput):
    """Output contract for ObserverAgent."""
    errors_found: bool = Field(default=False, description="Whether any errors were detected")
    errors: Dict[str, str] = Field(
        default_factory=dict,
        description="Map of node_id to error message"
    )


# =============================================================================
# Validation Helpers
# =============================================================================

def validate_input(inputs: dict, model_class: type) -> tuple:
    """Validate inputs against a Pydantic model.
    
    Parameters
    ----------
    inputs : dict
        Raw input dictionary from orchestrator.
    model_class : type
        Pydantic model class to validate against.
    
    Returns
    -------
    tuple
        (validated_model, None) on success, or (None, {"error": str}) on failure.
    """
    from pydantic import ValidationError
    
    try:
        validated = model_class(**inputs)
        return validated, None
    except ValidationError as e:
        error_msg = f"Input validation failed: {e}"
        return None, {"error": error_msg}


def create_output(model_class: type, **kwargs) -> dict:
    """Create a validated output dictionary.
    
    Parameters
    ----------
    model_class : type
        Pydantic output model class.
    **kwargs
        Output field values.
    
    Returns
    -------
    dict
        Validated output as dictionary.
    """
    output = model_class(**kwargs)
    return output.model_dump()
