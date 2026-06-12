from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException, UploadFile

from app.api.v1.endpoints.documents import upload_document


@pytest.mark.asyncio
async def test_upload_document_missing_filename_unit():
    """
    Manually invokes the endpoint to reach the filename validation line.
    This bypasses FastAPI's 422 validation to hit the 400 check in our code.
    """
    mock_file = MagicMock(spec=UploadFile)
    mock_file.filename = None  # Force the missing filename state

    mock_db = MagicMock()

    with pytest.raises(HTTPException) as excinfo:
        await upload_document(file=mock_file, current_user="test_user", db=mock_db)

    assert excinfo.value.status_code == 400
    assert "filename" in excinfo.value.detail.lower()


@pytest.mark.asyncio
async def test_upload_plan_artifact_unit():
    """
    Verifies that PLAN_ARTIFACT upload correctly bypasses the Celery pipeline
    and marks the job as COMPLETED immediately.
    """
    from app.models.document import DocumentType, JobStatus

    mock_file = AsyncMock(spec=UploadFile)
    mock_file.filename = "rubric.txt"
    mock_file.content_type = "text/plain"
    mock_file.read.return_value = b"test content"

    mock_db = MagicMock()
    mock_db.query.return_value.filter.return_value.first.return_value = None

    with (
        patch(
            "app.api.v1.endpoints.documents.storage_service.upload_file",
            new_callable=AsyncMock,
        ) as mock_upload,
        patch("app.api.v1.endpoints.documents.Document") as mock_doc_class,
        patch("app.api.v1.endpoints.documents.IngestionJob") as mock_job_class,
        patch(
            "app.api.v1.endpoints.documents.start_ingestion_pipeline"
        ) as mock_pipeline,
    ):
        mock_upload.return_value = "s3://bucket/test_user/rubric.txt"

        # Setup mock instances
        mock_doc = mock_doc_class.return_value
        mock_doc.id = 123

        mock_job = mock_job_class.return_value
        mock_job.id = 456

        response = await upload_document(
            file=mock_file,
            current_user="test_user",
            db=mock_db,
            doc_type=DocumentType.PLAN_ARTIFACT,
        )

    assert response["job_id"] == 456
    assert mock_job.status == JobStatus.COMPLETED
    mock_pipeline.assert_not_called()
    mock_db.commit.assert_called()
