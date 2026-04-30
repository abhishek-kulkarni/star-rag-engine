import asyncio
from datetime import UTC, datetime

from celery import chain  # type: ignore

from app.core.database import SessionLocal
from app.models.document import Document, DocumentChunk, IngestionJob, JobStatus
from app.services.llm_service import LLMService
from app.services.parser_service import parser_service
from app.services.storage_service import storage_service
from app.workers.celery_app import celery_app

# Lazy-loaded services to avoid module-level side effects during testing
_llm_service = None


def get_llm_service():
    global _llm_service
    if _llm_service is None:
        _llm_service = LLMService()
    return _llm_service


def update_job_status(job_id: int, status: JobStatus, error: str | None = None):
    """Utility to update IngestionJob status in the database."""
    with SessionLocal() as db:
        job = db.query(IngestionJob).filter(IngestionJob.id == job_id).first()
        if job:
            job.status = status
            if error:
                job.error_message = error

            if status == JobStatus.PARSING:
                job.parsed_at = datetime.now(UTC)
            elif status == JobStatus.CHUNKING:
                job.chunked_at = datetime.now(UTC)
            elif status == JobStatus.COMPLETED:
                job.completed_at = datetime.now(UTC)

            db.commit()


@celery_app.task(name="app.workers.tasks.parse_task")
def parse_task(data: dict):
    """Downloads PDF, extracts text, and saves text back to storage."""
    job_id = data["job_id"]
    document_id = data["document_id"]
    update_job_status(job_id, JobStatus.PARSING)

    try:
        with SessionLocal() as db:
            doc = db.query(Document).filter(Document.id == document_id).first()
            if not doc:
                raise ValueError(f"Document {document_id} not found")

            # Download raw PDF
            content = asyncio.run(storage_service.download_file(doc.storage_uri))

            # Extract text
            text = parser_service.parse_pdf(content)

            # Save extracted text to storage (Claim Check)
            text_filename = f"parsed_{document_id}.txt"
            text_uri = asyncio.run(
                storage_service.upload_file(
                    filename=text_filename,
                    content=text.encode("utf-8"),
                    user_id=str(doc.owner_id),
                    content_type="text/plain",
                )
            )

            return {
                "job_id": job_id,
                "document_id": document_id,
                "text_uri": text_uri,
                "owner_id": doc.owner_id,
            }
    except Exception as e:
        update_job_status(job_id, JobStatus.FAILED, str(e))
        raise


@celery_app.task(name="app.workers.tasks.chunk_task")
def chunk_task(data: dict):
    """Downloads text, chunks it, and persists chunks (no vectors yet)."""
    job_id = data["job_id"]
    document_id = data["document_id"]
    text_uri = data["text_uri"]
    update_job_status(job_id, JobStatus.CHUNKING)

    try:
        # Download extracted text
        text_bytes = asyncio.run(storage_service.download_file(text_uri))
        text = text_bytes.decode("utf-8")

        # Semantic splitting
        chunks = parser_service.split_text(text)

        with SessionLocal() as db:
            # Idempotency: Clear existing chunks for this document
            db.query(DocumentChunk).filter(
                DocumentChunk.document_id == document_id
            ).delete()

            # Bulk insert chunks with empty embeddings
            for i, chunk_text in enumerate(chunks):
                db_chunk = DocumentChunk(
                    document_id=document_id,
                    chunk_index=i,
                    text_content=chunk_text,
                    embedding=None,  # To be filled by next task
                )
                db.add(db_chunk)

            db.commit()

        return {"job_id": job_id, "document_id": document_id}
    except Exception as e:
        update_job_status(job_id, JobStatus.FAILED, str(e))
        raise


@celery_app.task(name="app.workers.tasks.embed_task")
def embed_task(data: dict):
    """Fetches chunks from DB, generates vectors, and updates DB."""
    job_id = data["job_id"]
    document_id = data["document_id"]
    update_job_status(job_id, JobStatus.EMBEDDING)

    try:
        llm = get_llm_service()
        with SessionLocal() as db:
            # Query chunks that need embedding
            chunks = (
                db.query(DocumentChunk)
                .filter(
                    DocumentChunk.document_id == document_id,
                    DocumentChunk.embedding.is_(None),
                )
                .all()
            )

            for chunk in chunks:
                # Get vector (sync wrapper for async LLM call)
                vector = asyncio.run(llm.get_embeddings(chunk.text_content))
                chunk.embedding = vector

            db.commit()

        update_job_status(job_id, JobStatus.COMPLETED)
        return True
    except Exception as e:
        update_job_status(job_id, JobStatus.FAILED, str(e))
        raise


def start_ingestion_pipeline(job_id: int, document_id: int):
    """Orchestrates the hardened ingestion chain."""
    payload = {"job_id": job_id, "document_id": document_id}
    pipeline = chain(parse_task.s(payload) | chunk_task.s() | embed_task.s())
    return pipeline.apply_async()
