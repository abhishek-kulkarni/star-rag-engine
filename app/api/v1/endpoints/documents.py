import logging
from typing import Annotated

import kombu.exceptions
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.core.logging import telemetry
from app.core.security import get_current_user
from app.models.document import Document, DocumentType, IngestionJob, JobStatus
from app.services.storage_service import storage_service
from app.workers.tasks import start_ingestion_pipeline

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/upload", status_code=status.HTTP_201_CREATED)
async def upload_document(
    file: Annotated[UploadFile, File(...)],
    current_user: Annotated[str, Depends(get_current_user)],
    db: Annotated[Session, Depends(get_db)],
    doc_type: Annotated[DocumentType, Form()] = DocumentType.STANDARD_DOC,
):
    """
    Handles multi-tenant document upload and triggers the ingestion pipeline.
    Includes Dual-Write Mitigation to handle broker connectivity failures.
    """
    # 1. Asynchronous upload to MinIO with user-prefixed storage
    # Validate filename existence and narrow type from str | None to str
    # to satisfy storage_service requirements.
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file must have a filename.",
        )

    content = await file.read()
    try:
        minio_uri = await storage_service.upload_file(
            filename=file.filename,
            content=content,
            user_id=current_user,
            content_type=file.content_type or "application/pdf",
        )
        logger.info(f"File {file.filename} uploaded to MinIO by {current_user}")
    except Exception as e:
        logger.error(f"MinIO upload failed for {file.filename}: {str(e)}")
        telemetry.storage_errors.labels(service="minio").inc()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Storage upload failed. Please try again later.",
        ) from e

    # 2. Database state initialization
    doc = Document(
        user_id=current_user,
        filename=file.filename,
        doc_type=doc_type,
        minio_raw_uri=minio_uri,
    )
    db.add(doc)
    db.flush()  # Extract the auto-incremented doc.id

    job = IngestionJob(document_id=doc.id, status=JobStatus.PENDING)
    db.add(job)
    db.commit()
    logger.info(
        f"DB record initialized for {file.filename} (DocID: {doc.id}, JobID: {job.id})"
    )

    # 3. Trigger Celery Pipeline with Dual-Write Mitigation (Only for Standard Docs)
    if doc_type == DocumentType.STANDARD_DOC:
        try:
            start_ingestion_pipeline(job.id, doc.id)
            logger.info(f"Ingestion pipeline triggered for JobID: {job.id}")
        except kombu.exceptions.OperationalError as e:
            # Broker (Redis) is down. Roll back job state to prevent "stuck" jobs.
            job.status = JobStatus.FAILED
            job.error_message = "Broker connection failed during ingestion trigger."
            db.commit()
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Ingestion service unavailable. Please try again later.",
            ) from e
    else:
        # Plan Artifacts skip the parsing/vectorization pipeline.
        # They are just stored in MinIO for reference/injection.
        job.status = JobStatus.COMPLETED
        db.commit()
        logger.info(f"Plan Artifact {file.filename} persisted (JobID: {job.id})")

    message = (
        "Upload successful. Ingestion started."
        if doc_type == DocumentType.STANDARD_DOC
        else "Plan Artifact uploaded successfully."
    )
    return {
        "document_id": doc.id,
        "job_id": job.id,
        "message": message,
    }


@router.get("/jobs/{job_id}")
async def get_job_status(
    job_id: int,
    current_user: Annotated[str, Depends(get_current_user)],
    db: Annotated[Session, Depends(get_db)],
):
    """Retrieves the status of an ingestion job, scoped to the current user."""
    job = (
        db
        .query(IngestionJob)
        .join(Document)
        .filter(IngestionJob.id == job_id, Document.user_id == current_user)
        .first()
    )

    if not job:
        raise HTTPException(status_code=404, detail="Ingestion job not found")

    return {
        "job_id": job.id,
        "status": job.status,
        "error": job.error_message,
        "created_at": job.created_at,
        "completed_at": job.completed_at,
    }


@router.get("/jobs")
async def list_jobs(
    current_user: Annotated[str, Depends(get_current_user)],
    db: Annotated[Session, Depends(get_db)],
):
    """Lists the last 5 ingestion jobs for the current user."""
    logger.info(f"User {current_user} requested ingestion job list")
    jobs = (
        db
        .query(IngestionJob)
        .join(Document)
        .filter(Document.user_id == current_user)
        .order_by(IngestionJob.created_at.desc())
        .limit(5)
        .all()
    )

    return [
        {
            "job_id": job.id,
            "document_id": job.document_id,
            "filename": job.document.filename,
            "status": job.status,
            "error": job.error_message,
            "created_at": job.created_at,
            "completed_at": job.completed_at,
        }
        for job in jobs
    ]


@router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_document(
    document_id: int,
    current_user: Annotated[str, Depends(get_current_user)],
    db: Annotated[Session, Depends(get_db)],
):
    """
    Synchronous cascade deletion (Right to be Forgotten).
    Purges file storage, DB records, and vector embeddings.
    """
    # 1. Verify ownership
    doc = (
        db
        .query(Document)
        .filter(Document.id == document_id, Document.user_id == current_user)
        .first()
    )

    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    # 2. Delete raw file from MinIO
    try:
        await storage_service.delete_file(doc.minio_raw_uri)
    except Exception as e:
        # CRITICAL: Failed to purge physical file. Log for manual intervention
        # but proceed with DB deletion to stop RAG engine from serving data.
        telemetry.storage_errors.labels(service="minio").inc()
        logger.critical(
            f"Orphaned file in storage: {doc.minio_raw_uri}. "
            f"Manual intervention required for compliance. Error: {str(e)}"
        )

    # 3. Synchronous DB Delete (CASCADE handles IngestionJob and DocumentChunks)
    db.delete(doc)
    db.commit()

    logger.info(
        f"Successfully purged document {document_id} and data for {current_user}"
    )

    return None
