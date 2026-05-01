from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.models.document import DocumentType, JobStatus

# --- Client Requests ---


class RAGQueryRequest(BaseModel):
    """Input contract for RAG retrieval and generation."""

    query: str = Field(..., min_length=1)
    include_plan_artifacts: bool = True


# --- API Responses ---


class IngestionJobResponse(BaseModel):
    """Detailed status of a document ingestion process."""

    id: int
    status: JobStatus
    error_message: str | None = None
    created_at: datetime
    parsed_at: datetime | None = None
    chunked_at: datetime | None = None
    completed_at: datetime | None = None

    model_config = ConfigDict(from_attributes=True)


class DocumentUploadResponse(BaseModel):
    """Initial response after a successful file upload and job creation."""

    document_id: int
    filename: str
    doc_type: DocumentType
    job: IngestionJobResponse
    message: str


class STARAnswerResponse(BaseModel):
    """
    Enforced structure for the LLM Orchestrator via Gemini response_schema.
    """

    situation: str
    task: str
    action: str
    result: str
    citations: list[int] = Field(
        description="Array of chunk IDs used to ground the answer"
    )

    model_config = ConfigDict(from_attributes=True)


class QueryResponse(BaseModel):
    """Wrapped RAG response with generated answer and source context."""

    answer: STARAnswerResponse
    source_nodes: list[dict[str, Any]] = []
