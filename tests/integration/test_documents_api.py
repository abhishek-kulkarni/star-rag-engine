from unittest.mock import AsyncMock, MagicMock, patch

import kombu.exceptions
import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.core.security import get_current_user
from app.main import app
from app.models.document import Document, IngestionJob, JobStatus

client = TestClient(app)

# --- Mocks & Fixtures ---


def override_get_current_user():
    return "test_user_123"


@pytest.fixture
def mock_db():
    """Provides a mocked database session."""
    session = MagicMock(spec=Session)
    app.dependency_overrides[get_db] = lambda: session
    app.dependency_overrides[get_current_user] = override_get_current_user
    yield session
    app.dependency_overrides.clear()


# --- Test Cases ---


def test_upload_document_success(mock_db):
    """Verify successful upload triggers the ingestion pipeline."""
    with (
        patch(
            "app.api.v1.endpoints.documents.storage_service.upload_file",
            new_callable=AsyncMock,
        ) as mock_upload,
        patch(
            "app.api.v1.endpoints.documents.start_ingestion_pipeline"
        ) as mock_pipeline,
    ):
        mock_upload.return_value = "minio://test_user_123/test.pdf"

        files = {"file": ("test.pdf", b"pdf content", "application/pdf")}
        response = client.post("/api/v1/documents/upload", files=files)

        assert response.status_code == 201
        data = response.json()
        assert "document_id" in data
        assert "job_id" in data

        # Verify DB calls
        assert mock_db.add.call_count == 2  # Document and IngestionJob
        assert mock_db.commit.called
        mock_pipeline.assert_called_once()


def test_upload_document_missing_filename(mock_db):
    """Verify 422 error when filename is missing (handled by FastAPI)."""
    files = {"file": (None, b"content", "application/pdf")}
    response = client.post("/api/v1/documents/upload", files=files)
    assert response.status_code == 422


def test_upload_document_storage_failure(mock_db):
    """Verify 500 error when storage service fails."""
    with patch(
        "app.api.v1.endpoints.documents.storage_service.upload_file",
        side_effect=Exception("Storage full"),
    ):
        files = {"file": ("test.pdf", b"content", "application/pdf")}
        response = client.post("/api/v1/documents/upload", files=files)
        assert response.status_code == 500
        assert "Storage upload failed" in response.json()["detail"]


def test_upload_document_broker_failure(mock_db):
    """Verify 503 error and job rollback when Celery broker is down."""
    with (
        patch(
            "app.api.v1.endpoints.documents.storage_service.upload_file",
            new_callable=AsyncMock,
        ) as mock_upload,
        patch(
            "app.api.v1.endpoints.documents.start_ingestion_pipeline",
            side_effect=kombu.exceptions.OperationalError("Broker down"),
        ),
    ):
        mock_upload.return_value = "minio://test.pdf"
        files = {"file": ("test.pdf", b"content", "application/pdf")}

        response = client.post("/api/v1/documents/upload", files=files)

        assert response.status_code == 503

        # Verify job status was marked as FAILED via the mock object
        # The second call to add() is the IngestionJob
        job = mock_db.add.call_args_list[1][0][0]
        assert job.status == JobStatus.FAILED
        assert "Broker connection failed" in job.error_message


def test_get_job_status_success(mock_db):
    """Verify retrieving job status for own document."""
    mock_job = MagicMock(spec=IngestionJob)
    mock_job.id = 1
    mock_job.status = JobStatus.PARSING
    mock_job.error_message = None
    mock_job.created_at = None
    mock_job.completed_at = None

    query_mock = mock_db.query.return_value
    query_mock.join.return_value.filter.return_value.first.return_value = mock_job

    response = client.get("/api/v1/documents/jobs/1")
    assert response.status_code == 200
    assert response.json()["status"] == "PARSING"


def test_get_job_status_not_found(mock_db):
    """Verify 404 when job is missing or belongs to another user."""
    query_mock = mock_db.query.return_value
    query_mock.join.return_value.filter.return_value.first.return_value = None

    response = client.get("/api/v1/documents/jobs/999")
    assert response.status_code == 404


def test_delete_document_success(mock_db):
    """Verify successful cascade deletion."""
    mock_doc = MagicMock(spec=Document)
    mock_doc.id = 1
    mock_doc.minio_raw_uri = "minio://test"

    mock_db.query.return_value.filter.return_value.first.return_value = mock_doc

    with patch(
        "app.api.v1.endpoints.documents.storage_service.delete_file",
        new_callable=AsyncMock,
    ) as mock_delete:
        response = client.delete("/api/v1/documents/1")
        assert response.status_code == 204

        mock_delete.assert_called_once_with("minio://test")
        mock_db.delete.assert_called_once_with(mock_doc)


def test_delete_document_not_found(mock_db):
    """Verify 404 when document is missing."""
    mock_db.query.return_value.filter.return_value.first.return_value = None
    response = client.delete("/api/v1/documents/999")
    assert response.status_code == 404


def test_delete_document_storage_resilience(mock_db):
    """Verify DB delete proceeds even if storage delete fails."""
    mock_doc = MagicMock(spec=Document)
    mock_doc.minio_raw_uri = "minio://fail"

    mock_db.query.return_value.filter.return_value.first.return_value = mock_doc

    with patch(
        "app.api.v1.endpoints.documents.storage_service.delete_file",
        side_effect=Exception("MinIO down"),
    ):
        response = client.delete("/api/v1/documents/1")
        assert response.status_code == 204
        # Verify DB delete was still called
        mock_db.delete.assert_called_once_with(mock_doc)


def test_get_jobs_list_success(mock_db):
    """Verify retrieving list of jobs for the current user."""
    mock_job = MagicMock(spec=IngestionJob)
    mock_job.id = 1
    mock_job.status = JobStatus.COMPLETED
    mock_job.document = MagicMock(spec=Document)
    mock_job.document.filename = "test.pdf"
    mock_job.created_at = None
    mock_job.completed_at = None
    mock_job.error_message = None

    query_mock = mock_db.query.return_value
    (
        query_mock.join.return_value.filter.return_value.order_by.return_value.limit.return_value.all.return_value
    ) = [mock_job]

    response = client.get("/api/v1/documents/jobs")
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 1
    assert data[0]["job_id"] == 1
    assert data[0]["filename"] == "test.pdf"
    assert data[0]["status"] == "COMPLETED"
