import logging
from datetime import UTC, datetime

from celery import Task, chain  # type: ignore

from app.config.settings import settings
from app.core.database import SessionLocal, ensure_user_partition
from app.core.logging import telemetry
from app.models.document import Document, DocumentChunk, IngestionJob, JobStatus
from app.services.llm_service import LLMService
from app.services.parser_service import parser_service
from app.services.storage_service import storage_service
from app.workers.celery_app import celery_app

logger = logging.getLogger(__name__)

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


class BaseIngestionTask(Task):
    """
    Base task for all ingestion stages.
    Automatically handles DB updates and DLQ routing on failure.
    """

    def on_failure(self, exc, task_id, args, kwargs, einfo):
        # Determine job_id from args
        # In our chain, args[0] is usually the dictionary containing job_id
        job_id = None
        if args and isinstance(args[0], dict):
            job_id = args[0].get("job_id")
        elif args and isinstance(args[0], int):
            # Fallback for direct job_id passing
            job_id = args[0]

        if job_id:
            update_job_status(job_id, JobStatus.FAILED, str(exc))

        # Quarantine the failed task payload in the DLQ for manual inspection
        self.apply_async(
            args=args, kwargs=kwargs, queue=settings.CELERY_DLQ_NAME, priority=0
        )


@celery_app.task(
    name="app.workers.tasks.parse_task",
    base=BaseIngestionTask,
    queue=settings.CELERY_INGESTION_QUEUE,
)
def parse_task(data: dict):
    """Downloads document, extracts text, and saves text back to storage."""
    job_id = data["job_id"]
    document_id = data["document_id"]
    logger.info(
        f"[Ingestion] parse_task started (JobID: {job_id}, DocID: {document_id})"
    )
    update_job_status(job_id, JobStatus.PARSING)

    try:
        with SessionLocal() as db:
            doc = db.query(Document).filter(Document.id == document_id).first()
            if not doc:
                raise ValueError(f"Document {document_id} not found")

            # Download raw document
            try:
                content = storage_service.download_file_sync(doc.minio_raw_uri)
                logger.info(f"[Ingestion] Document downloaded from {doc.minio_raw_uri}")
            except Exception:
                telemetry.storage_errors.labels(service="minio").inc()
                raise

            # Extract text
            try:
                text = parser_service.parse(content, doc.filename)
                logger.info(
                    f"[Ingestion] {doc.filename} text extracted ({len(text)} chars)"
                )
            except Exception:
                telemetry.processing_errors.labels(stage="parse").inc()
                raise

            # Save extracted text to storage (Claim Check)
            try:
                text_filename = f"parsed_{document_id}.txt"
                text_uri = storage_service.upload_file_sync(
                    filename=text_filename,
                    content=text.encode("utf-8"),
                    user_id=str(doc.user_id),
                    content_type="text/plain",
                )
                logger.info(f"[Ingestion] Extracted text saved to {text_uri}")
            except Exception:
                telemetry.storage_errors.labels(service="minio").inc()
                raise

            return {
                "job_id": job_id,
                "document_id": document_id,
                "text_uri": text_uri,
                "user_id": doc.user_id,
            }
    except Exception as e:
        logger.error(f"[Ingestion] parse_task FAILED for JobID: {job_id}: {str(e)}")
        raise


@celery_app.task(
    name="app.workers.tasks.chunk_task",
    base=BaseIngestionTask,
    queue=settings.CELERY_INGESTION_QUEUE,
)
def chunk_task(data: dict):
    """Downloads text, chunks it, and persists chunks (no vectors yet)."""
    job_id = data["job_id"]
    document_id = data["document_id"]
    text_uri = data["text_uri"]
    logger.info(f"[Ingestion] Starting chunk_task for JobID: {job_id}")
    update_job_status(job_id, JobStatus.CHUNKING)

    try:
        # Download extracted text
        try:
            text_bytes = storage_service.download_file_sync(text_uri)
            text = text_bytes.decode("utf-8")
        except Exception:
            telemetry.storage_errors.labels(service="minio").inc()
            raise

        # Semantic splitting
        try:
            chunks = parser_service.split_text(text)
            logger.info(f"[Ingestion] Split into {len(chunks)} semantic chunks")
        except Exception:
            telemetry.processing_errors.labels(stage="chunk").inc()
            raise

        try:
            with SessionLocal() as db:
                # Ensure the user partition exists before inserting
                user_id = data["user_id"]
                ensure_user_partition(db, user_id)

                # Idempotency: Clear existing chunks for this document
                # We include user_id to ensure partition pruning.
                db.query(DocumentChunk).filter(
                    DocumentChunk.document_id == document_id,
                    DocumentChunk.user_id == user_id,
                ).delete()

                # Bulk insert chunks with empty embeddings
                for i, chunk_text in enumerate(chunks):
                    db_chunk = DocumentChunk(
                        document_id=document_id,
                        user_id=user_id,
                        chunk_index=i,
                        text_content=chunk_text,
                        embedding=None,  # To be filled by next task
                    )
                    db.add(db_chunk)

                db.commit()
                logger.info(
                    f"[Ingestion] {len(chunks)} chunks persisted (DocID: {document_id})"
                )
        except Exception:
            telemetry.storage_errors.labels(service="postgres").inc()
            raise

        return {"job_id": job_id, "document_id": document_id, "user_id": user_id}
    except Exception as e:
        logger.error(f"[Ingestion] chunk_task FAILED for JobID: {job_id}: {str(e)}")
        raise


@celery_app.task(
    bind=True,
    name="app.workers.tasks.embed_task",
    base=BaseIngestionTask,
    queue=settings.CELERY_INGESTION_QUEUE,
    max_retries=10,
)
def embed_task(self, data: dict):
    """Fetches chunks from DB, generates vectors, and updates DB."""
    job_id = data["job_id"]
    document_id = data["document_id"]
    logger.info(f"[Ingestion] Starting embed_task for JobID: {job_id}")
    update_job_status(job_id, JobStatus.EMBEDDING)

    try:
        # embed_task receives user_id from the chain (passed from parse_task ->
        # chunk_task)
        user_id = data.get("user_id")
        llm = get_llm_service()
        try:
            with SessionLocal() as db:
                # Query chunks that need embedding
                # We include user_id to ensure partition pruning.
                query = db.query(DocumentChunk).filter(
                    DocumentChunk.document_id == document_id,
                    DocumentChunk.embedding.is_(None),
                )

                if user_id:
                    query = query.filter(DocumentChunk.user_id == user_id)

                chunks = query.all()

                if chunks:
                    logger.info(f"[Ingestion] Embedding {len(chunks)} chunks...")
                    # Batch processing to avoid payload size limits.
                    batch_size = 10

                    for i in range(0, len(chunks), batch_size):
                        batch_chunks = chunks[i : i + batch_size]
                        batch_texts = [c.text_content for c in batch_chunks]

                        try:
                            vectors = llm.get_embeddings_batch_sync(batch_texts)

                            # Guard against the API dropping vectors silently
                            if len(vectors) != len(batch_texts):
                                raise ValueError(
                                    f"API returned {len(vectors)} vectors "
                                    f"for {len(batch_texts)} chunks."
                                )

                        except Exception as e:
                            logger.warning(
                                f"[Ingestion] Batch failed "
                                f"(Retry {self.request.retries}): {str(e)}"
                            )
                            telemetry.llm_errors.labels(model="embed").inc()
                            # Yield the worker back to the pool with exponential backoff
                            # Formula: 2^retries * 5s (e.g. 5s, 10s, 20s...)
                            backoff = (2**self.request.retries) * 5
                            raise self.retry(exc=e, countdown=backoff) from e

                        # Zip the returned vectors back to the SQLAlchemy objects
                        for chunk, vector in zip(batch_chunks, vectors, strict=True):
                            chunk.embedding = vector

                        # Commit incrementally! If we hit a 429 on the next batch,
                        # the successful chunks are saved and won't be re-processed.
                        db.commit()
                        logger.info(
                            f"[Ingestion] Batch {i // batch_size + 1} completed"
                        )

        except Exception as e:
            # Avoid double-counting errors already tracked by telemetry
            if "telemetry" not in str(e) and not isinstance(e, self.retry.__class__):
                logger.error(
                    f"[Ingestion] embed_task CRITICAL failure for JobID: {job_id}: {e}"
                )
            raise

        update_job_status(job_id, JobStatus.COMPLETED)
        logger.info(f"[Ingestion] PIPELINE COMPLETED for JobID: {job_id}")
        return True
    except Exception:
        raise


def start_ingestion_pipeline(job_id: int, document_id: int):
    """Orchestrates the hardened ingestion chain."""
    payload = {"job_id": job_id, "document_id": document_id}
    pipeline = chain(parse_task.s(payload) | chunk_task.s() | embed_task.s())
    return pipeline.apply_async(queue="ingestion")
