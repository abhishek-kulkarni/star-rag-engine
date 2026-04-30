from unittest.mock import MagicMock

import pytest

from app.services.storage_service import StorageService


def test_storage_service_initialization():
    service = StorageService()
    assert service.client is not None


@pytest.mark.asyncio
async def test_upload_file(monkeypatch):
    service = StorageService()
    service.client = MagicMock()

    mock_data = b"test content"
    uri = await service.upload_file(
        "test.pdf", mock_data, "user_123", content_type="application/pdf"
    )

    assert "user_123/test.pdf" in uri
    # Check that put_object was called with the correct content_type
    # (last positional arg)
    args, kwargs = service.client.put_object.call_args
    assert args[4] == "application/pdf"


@pytest.mark.asyncio
async def test_delete_file(monkeypatch):
    service = StorageService()
    service.client = MagicMock()

    uri = "minio://star-rag-documents/user_123/test.pdf"
    await service.delete_file(uri)

    service.client.remove_object.assert_called_once_with(
        "star-rag-documents", "user_123/test.pdf"
    )


@pytest.mark.asyncio
async def test_delete_file_invalid_uri(monkeypatch):
    service = StorageService()
    service.client = MagicMock()

    # URI from another bucket or invalid format
    uri = "minio://other-bucket/user_123/test.pdf"
    await service.delete_file(uri)

    service.client.remove_object.assert_not_called()


def test_ensure_bucket_exists_creates_if_missing(monkeypatch):
    # Mock Minio class to control behavior during __init__
    mock_minio = MagicMock()
    mock_minio.bucket_exists.return_value = False
    monkeypatch.setattr(
        "app.services.storage_service.Minio", lambda *a, **k: mock_minio
    )

    # This will trigger __init__ -> _ensure_bucket_exists
    StorageService()

    mock_minio.make_bucket.assert_called_once_with("star-rag-documents")
