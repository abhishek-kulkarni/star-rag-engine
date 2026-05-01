from unittest.mock import MagicMock, patch

import pytest

from app.config.settings import settings
from app.services.storage_service import StorageService


@pytest.fixture
def storage_service():
    with patch("app.services.storage_service.Minio"):
        service = StorageService()
        return service


@pytest.mark.asyncio
async def test_upload_file(storage_service):
    """Verify file upload offloads to thread pool."""
    with patch("anyio.to_thread.run_sync") as mock_run:
        uri = await storage_service.upload_file(
            filename="test.pdf", content=b"data", user_id="user1"
        )
        assert uri == f"minio://{settings.MINIO_BUCKET_NAME}/user1/test.pdf"
        mock_run.assert_called_once()


@pytest.mark.asyncio
async def test_download_file(storage_service):
    """Verify file download retrieves bytes."""
    mock_response = MagicMock()
    mock_response.read.return_value = b"file data"

    with patch("anyio.to_thread.run_sync", return_value=mock_response):
        content = await storage_service.download_file(
            f"minio://{settings.MINIO_BUCKET_NAME}/user1/test.pdf"
        )
        assert content == b"file data"
        mock_response.read.assert_called_once()


@pytest.mark.asyncio
async def test_download_file_invalid_uri(storage_service):
    """Verify download raises error for invalid URI."""
    with pytest.raises(ValueError, match="Invalid MinIO URI"):
        await storage_service.download_file("invalid://uri")


@pytest.mark.asyncio
async def test_delete_file(storage_service):
    """Verify file deletion logic."""
    with patch("anyio.to_thread.run_sync") as mock_run:
        await storage_service.delete_file(
            f"minio://{settings.MINIO_BUCKET_NAME}/user1/test.pdf"
        )
        mock_run.assert_called_once()


@pytest.mark.asyncio
async def test_delete_file_invalid_uri(storage_service):
    """Verify delete ignores invalid URIs silently."""
    with patch("anyio.to_thread.run_sync") as mock_run:
        await storage_service.delete_file("invalid://uri")
        mock_run.assert_not_called()


def test_bucket_creation():
    """Verify bucket is created if missing."""
    with patch("app.services.storage_service.Minio") as mock_minio:
        client = mock_minio.return_value
        client.bucket_exists.return_value = False
        StorageService()
        client.make_bucket.assert_called_once_with(settings.MINIO_BUCKET_NAME)
