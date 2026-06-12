from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.core.security import get_current_user
from app.main import app
from app.models.document import Document, DocumentType

client = TestClient(app)


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


def test_upload_duplicate_filename_no_overwrite(mock_db):
    """
    Verify upload fails with 409 when filename already exists
    and overwrite is False.
    """
    existing_doc = Document(
        id=1,
        user_id="test_user_123",
        filename="duplicate.pdf",
        doc_type=DocumentType.STANDARD_DOC,
        minio_raw_uri="minio://starrag/test_user_123/duplicate.pdf",
        content_hash="some_other_hash",
    )

    # Use a single query mock so side_effect is shared across queries
    query_mock = MagicMock()
    query_mock.filter.return_value.first.side_effect = [None, existing_doc]
    mock_db.query.return_value = query_mock

    files = {"file": ("duplicate.pdf", b"some pdf content", "application/pdf")}
    data = {"overwrite": "false"}

    response = client.post("/api/v1/documents/upload", files=files, data=data)

    assert response.status_code == 409
    detail = response.json()["detail"]
    assert "Document with filename 'duplicate.pdf' already exists" in detail


def test_upload_duplicate_content_no_overwrite(mock_db):
    """
    Verify upload fails with 409 when file content hash already exists
    and overwrite is False.
    """
    target_hash = "b686300438137351ff5606e3ea86c23b2075a4092b3a0f7e4de8892f39872c08"
    existing_doc = Document(
        id=1,
        user_id="test_user_123",
        filename="original.pdf",
        doc_type=DocumentType.STANDARD_DOC,
        minio_raw_uri="minio://starrag/test_user_123/original.pdf",
        content_hash=target_hash,
    )

    query_mock = MagicMock()
    query_mock.filter.return_value.first.return_value = existing_doc
    mock_db.query.return_value = query_mock

    files = {"file": ("different_name.pdf", b"some pdf content", "application/pdf")}
    data = {"overwrite": "false"}

    response = client.post("/api/v1/documents/upload", files=files, data=data)

    assert response.status_code == 409
    detail = response.json()["detail"]
    assert "identical to an existing document named 'original.pdf'" in detail


def test_upload_overwrite_success(mock_db):
    """
    Verify upload succeeds and prunes existing file and DB record
    when overwrite is True.
    """
    target_hash = "b686300438137351ff5606e3ea86c23b2075a4092b3a0f7e4de8892f39872c08"
    existing_doc = Document(
        id=1,
        user_id="test_user_123",
        filename="duplicate.pdf",
        doc_type=DocumentType.STANDARD_DOC,
        minio_raw_uri="minio://starrag/test_user_123/duplicate.pdf",
        content_hash=target_hash,
    )

    query_mock = MagicMock()
    query_mock.filter.return_value.first.return_value = existing_doc
    query_mock.filter.return_value.all.return_value = [existing_doc]
    mock_db.query.return_value = query_mock

    with (
        patch(
            "app.api.v1.endpoints.documents.storage_service.upload_file",
            new_callable=AsyncMock,
        ) as mock_upload,
        patch(
            "app.api.v1.endpoints.documents.storage_service.delete_file",
            new_callable=AsyncMock,
        ) as mock_delete,
        patch(
            "app.api.v1.endpoints.documents.start_ingestion_pipeline"
        ) as mock_pipeline,
    ):
        mock_upload.return_value = "minio://starrag/test_user_123/duplicate.pdf"

        files = {"file": ("duplicate.pdf", b"some pdf content", "application/pdf")}
        data = {"overwrite": "true"}

        response = client.post("/api/v1/documents/upload", files=files, data=data)

        assert response.status_code == 201
        assert response.json()["message"] == ("Upload successful. Ingestion started.")

        # Verify old file was deleted
        mock_delete.assert_called_once_with(
            "minio://starrag/test_user_123/duplicate.pdf"
        )
        # Verify old DB doc was deleted
        mock_db.delete.assert_called_once_with(existing_doc)
        # Verify pipeline triggered for new doc
        mock_pipeline.assert_called_once()


def test_upload_overwrite_minio_delete_resilience(mock_db):
    """
    Verify DB delete still proceeds and upload succeeds even if
    old MinIO file delete fails.
    """
    target_hash = "b686300438137351ff5606e3ea86c23b2075a4092b3a0f7e4de8892f39872c08"
    existing_doc = Document(
        id=1,
        user_id="test_user_123",
        filename="duplicate.pdf",
        doc_type=DocumentType.STANDARD_DOC,
        minio_raw_uri="minio://starrag/test_user_123/duplicate.pdf",
        content_hash=target_hash,
    )

    query_mock = MagicMock()
    query_mock.filter.return_value.first.return_value = existing_doc
    query_mock.filter.return_value.all.return_value = [existing_doc]
    mock_db.query.return_value = query_mock

    with (
        patch(
            "app.api.v1.endpoints.documents.storage_service.upload_file",
            new_callable=AsyncMock,
        ) as mock_upload,
        patch(
            "app.api.v1.endpoints.documents.storage_service.delete_file",
            side_effect=Exception("MinIO connection reset"),
        ),
        patch(
            "app.api.v1.endpoints.documents.start_ingestion_pipeline"
        ) as mock_pipeline,
    ):
        mock_upload.return_value = "minio://starrag/test_user_123/duplicate.pdf"

        files = {"file": ("duplicate.pdf", b"some pdf content", "application/pdf")}
        data = {"overwrite": "true"}

        response = client.post("/api/v1/documents/upload", files=files, data=data)

        assert response.status_code == 201
        # Verify DB delete was still called despite MinIO exception
        mock_db.delete.assert_called_once_with(existing_doc)
        mock_pipeline.assert_called_once()
