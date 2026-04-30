import io

import anyio
from minio import Minio

from app.config.settings import settings


class StorageService:
    """
    Handles unstructured file persistence using MinIO (S3-compatible).
    Provides a unified interface for storing and retrieving raw technical documents
    under user-specific prefixes to enforce multi-tenancy at the storage layer.
    """

    def __init__(self):
        self.client = Minio(
            settings.MINIO_ENDPOINT,
            access_key=settings.MINIO_ROOT_USER,
            secret_key=settings.MINIO_ROOT_PASSWORD,
            secure=settings.MINIO_SECURE,
        )
        self.bucket_name = "star-rag-documents"
        self._ensure_bucket_exists()

    def _ensure_bucket_exists(self):
        """
        Idempotent check to ensure the application bucket is available on start.
        """
        if not self.client.bucket_exists(self.bucket_name):
            self.client.make_bucket(self.bucket_name)

    async def upload_file(
        self,
        filename: str,
        content: bytes,
        user_id: str,
        content_type: str = "application/octet-stream",
    ) -> str:
        """
        Uploads a file to MinIO under a user-specific prefix.
        Returns the MinIO URI.
        """
        object_name = f"{user_id}/{filename}"
        content_stream = io.BytesIO(content)

        # Offload synchronous MinIO I/O to a separate thread pool
        await anyio.to_thread.run_sync(
            self.client.put_object,
            self.bucket_name,
            object_name,
            content_stream,
            len(content),
            content_type,
        )

        return f"minio://{self.bucket_name}/{object_name}"

    async def delete_file(self, minio_uri: str):
        """
        Deletes a file from MinIO given its URI.
        """
        if not minio_uri.startswith(f"minio://{self.bucket_name}/"):
            return

        object_name = minio_uri.replace(f"minio://{self.bucket_name}/", "")

        # Offload synchronous MinIO I/O to a separate thread pool
        await anyio.to_thread.run_sync(
            self.client.remove_object, self.bucket_name, object_name
        )


storage_service = StorageService()
