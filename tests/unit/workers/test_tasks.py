from unittest.mock import MagicMock, patch

import pytest

from app.config.settings import settings
from app.models.document import Document, DocumentChunk, IngestionJob, JobStatus
from app.workers.tasks import (
    BaseIngestionTask,
    chunk_task,
    embed_task,
    get_llm_service,
    parse_task,
    start_ingestion_pipeline,
)


def test_base_task_on_failure(mock_db_session):
    """Verify BaseIngestionTask updates job status on failure."""
    task = BaseIngestionTask()
    exc = Exception("Task failed")
    args = [{"job_id": 1}]

    with (
        patch("app.workers.tasks.update_job_status") as mock_update,
        patch.object(BaseIngestionTask, "apply_async") as mock_apply,
    ):
        task.on_failure(exc, "task-id", args, {}, None)
        mock_update.assert_called_once_with(1, JobStatus.FAILED, "Task failed")
        mock_apply.assert_called_once_with(
            args=args, kwargs={}, queue=settings.CELERY_DLQ_NAME, priority=0
        )


def test_base_task_on_failure_direct_id(mock_db_session):
    """Verify BaseIngestionTask handles direct integer job_id in args."""
    task = BaseIngestionTask()
    exc = Exception("Direct fail")
    args = [5]  # Direct int ID

    with (
        patch("app.workers.tasks.update_job_status") as mock_update,
        patch.object(BaseIngestionTask, "apply_async") as mock_apply,
    ):
        task.on_failure(exc, "task-id", args, {}, None)
        mock_update.assert_called_once_with(5, JobStatus.FAILED, "Direct fail")
        mock_apply.assert_called_once()


def test_get_llm_service_lazy_init(monkeypatch):
    """Verify LLM service lazy initialization."""
    with patch("app.workers.tasks.LLMService") as mock_class:
        import app.workers.tasks

        monkeypatch.setattr(app.workers.tasks, "_llm_service", None)
        service = get_llm_service()
        assert service is not None
        mock_class.assert_called_once()


def test_update_job_status_branches(mock_db_session):
    """Verify all branches of update_job_status (timestamps)."""
    from app.workers.tasks import update_job_status

    # Using spec=IngestionJob to ensure attribute safety
    mock_job = MagicMock(spec=IngestionJob)
    mock_db_session.query.return_value.filter.return_value.first.return_value = mock_job

    update_job_status(1, JobStatus.PARSING)
    assert mock_job.status == JobStatus.PARSING
    update_job_status(1, JobStatus.CHUNKING)
    assert mock_job.status == JobStatus.CHUNKING
    update_job_status(1, JobStatus.COMPLETED)
    assert mock_job.status == JobStatus.COMPLETED
    update_job_status(1, JobStatus.FAILED, error="Some error")
    assert mock_job.error_message == "Some error"


def test_parse_task_doc_not_found(mock_db_session):
    """Verify parse_task handles missing document."""
    data = {"job_id": 1, "document_id": 999}
    mock_db_session.query.return_value.filter.return_value.first.return_value = None
    with patch("app.workers.tasks.update_job_status"):
        with pytest.raises(ValueError, match="Document 999 not found"):
            parse_task(data)


def test_parse_task_logic(mock_db_session):
    """Verify parse_task downloads PDF, extracts text, and uploads it."""
    data = {"job_id": 1, "document_id": 10}
    # Using spec=Document to catch schema mismatches
    mock_doc = MagicMock(spec=Document)
    mock_doc.id = 10
    mock_doc.user_id = 99
    mock_doc.minio_raw_uri = "minio://pdf"

    mock_db_session.query.return_value.filter.return_value.first.return_value = mock_doc

    with (
        patch("app.workers.tasks.storage_service.download_file_sync") as mock_download,
        patch("app.workers.tasks.parser_service.parse") as mock_parse,
        patch("app.workers.tasks.storage_service.upload_file_sync") as mock_upload,
    ):
        mock_download.return_value = b"pdf bytes"
        mock_upload.return_value = "minio://text"
        mock_parse.return_value = "extracted text"
        result = parse_task(data)
        assert result["text_uri"] == "minio://text"
        mock_db_session.commit.assert_called()


def test_chunk_task_logic(mock_db_session):
    """Verify chunk_task downloads text, splits it, and saves to DB."""
    data = {
        "job_id": 1,
        "document_id": 10,
        "text_uri": "minio://text",
        "user_id": "user_123",
    }
    with (
        patch("app.workers.tasks.storage_service.download_file_sync") as mock_download,
        patch("app.workers.tasks.parser_service.split_text") as mock_split,
        patch("app.workers.tasks.ensure_user_partition") as mock_ensure,
    ):
        mock_download.return_value = b"text bytes"
        mock_split.return_value = ["chunk 1", "chunk 2"]
        result = chunk_task(data)
        assert result["document_id"] == 10
        assert result["user_id"] == "user_123"
        mock_ensure.assert_called_once()
        assert mock_db_session.add.call_count == 2
        mock_db_session.commit.assert_called()


def test_embed_task_logic(mock_db_session):
    """Verify embed_task fetches null-embedding chunks and fills them."""
    data = {"job_id": 1, "document_id": 10, "user_id": "user_123"}
    # Using spec=DocumentChunk for schema safety
    chunk1 = MagicMock(spec=DocumentChunk)
    chunk1.text_content = "text 1"
    chunk1.embedding = None

    # The query now has two filter calls when user_id is present
    mock_db_session.query.return_value.filter.return_value.all.return_value = [chunk1]
    (
        mock_db_session.query.return_value.filter.return_value.filter.return_value.all.return_value
    ) = [chunk1]

    mock_llm = MagicMock()
    mock_llm.get_embeddings_batch_sync.return_value = [[0.1]]

    with (
        patch("app.workers.tasks.get_llm_service", return_value=mock_llm),
    ):
        result = embed_task(data)
        assert result is True
        assert chunk1.embedding == [0.1]
        mock_db_session.commit.assert_called()
        mock_llm.get_embeddings_batch_sync.assert_called_once()


def test_parse_task_failure(mock_db_session):
    """Verify parse_task raises exception on DB failure."""
    data = {"job_id": 1, "document_id": 10}
    with patch("app.workers.tasks.update_job_status"):
        mock_db_session.query.return_value.filter.return_value.first.side_effect = (
            Exception("DB error")
        )
        with pytest.raises(Exception, match="DB error"):
            parse_task(data)


def test_chunk_task_failure(mock_db_session):
    """Verify chunk_task raises exception on storage failure."""
    data = {
        "job_id": 1,
        "document_id": 10,
        "text_uri": "minio://text",
        "user_id": "user_123",
    }
    with patch("app.workers.tasks.update_job_status"):
        with patch(
            "app.workers.tasks.storage_service.download_file_sync",
            side_effect=Exception("Storage error"),
        ):
            with pytest.raises(Exception, match="Storage error"):
                chunk_task(data)


def test_embed_task_failure(mock_db_session):
    """Verify embed_task raises exception on LLM failure."""
    data = {"job_id": 1, "document_id": 10, "user_id": "user_123"}
    # The query now has two filter calls when user_id is present
    mock_db_session.query.return_value.filter.return_value.all.return_value = [
        MagicMock(spec=DocumentChunk)
    ]
    (
        mock_db_session.query.return_value.filter.return_value.filter.return_value.all.return_value
    ) = [MagicMock(spec=DocumentChunk)]
    with patch("app.workers.tasks.update_job_status"):
        mock_llm = MagicMock()
        mock_llm.get_embeddings_batch_sync.side_effect = Exception("LLM error")
        with patch("app.workers.tasks.get_llm_service", return_value=mock_llm):
            with pytest.raises(Exception, match="LLM error"):
                embed_task(data)


def test_parse_task_download_failure(mock_db_session):
    """Verify parse_task handles storage download error."""
    data = {"job_id": 1, "document_id": 10}
    mock_doc = MagicMock(spec=Document)
    mock_doc.minio_raw_uri = "minio://pdf"
    mock_db_session.query.return_value.filter.return_value.first.return_value = mock_doc

    with patch("app.workers.tasks.update_job_status"):
        with patch(
            "app.workers.tasks.storage_service.download_file_sync",
            side_effect=Exception("Download failed"),
        ):
            with pytest.raises(Exception, match="Download failed"):
                parse_task(data)


def test_parse_task_extract_failure(mock_db_session):
    """Verify parse_task handles text extraction error."""
    data = {"job_id": 1, "document_id": 10}
    mock_doc = MagicMock(spec=Document)
    mock_doc.minio_raw_uri = "minio://pdf"
    mock_db_session.query.return_value.filter.return_value.first.return_value = mock_doc

    with (
        patch("app.workers.tasks.update_job_status"),
        patch(
            "app.workers.tasks.storage_service.download_file_sync",
            return_value=b"pdf bytes",
        ),
        patch(
            "app.workers.tasks.parser_service.parse",
            side_effect=Exception("Parse failed"),
        ),
    ):
        with pytest.raises(Exception, match="Parse failed"):
            parse_task(data)


def test_parse_task_upload_failure(mock_db_session):
    """Verify parse_task handles storage upload error."""
    data = {"job_id": 1, "document_id": 10}
    mock_doc = MagicMock(spec=Document)
    mock_doc.minio_raw_uri = "minio://pdf"
    mock_doc.user_id = 99
    mock_db_session.query.return_value.filter.return_value.first.return_value = mock_doc

    with (
        patch("app.workers.tasks.update_job_status"),
        patch(
            "app.workers.tasks.storage_service.download_file_sync",
            return_value=b"pdf bytes",
        ),
        patch("app.workers.tasks.parser_service.parse", return_value="text"),
        patch(
            "app.workers.tasks.storage_service.upload_file_sync",
            side_effect=Exception("Upload failed"),
        ),
    ):
        with pytest.raises(Exception, match="Upload failed"):
            parse_task(data)


def test_chunk_task_split_failure(mock_db_session):
    """Verify chunk_task handles semantic splitting error."""
    data = {
        "job_id": 1,
        "document_id": 10,
        "text_uri": "minio://text",
        "user_id": "user_123",
    }
    with (
        patch("app.workers.tasks.update_job_status"),
        patch(
            "app.workers.tasks.storage_service.download_file_sync",
            return_value=b"text bytes",
        ),
        patch(
            "app.workers.tasks.parser_service.split_text",
            side_effect=Exception("Split failed"),
        ),
    ):
        with pytest.raises(Exception, match="Split failed"):
            chunk_task(data)


def test_chunk_task_db_insert_failure(mock_db_session):
    """Verify chunk_task handles database insertion error."""
    data = {
        "job_id": 1,
        "document_id": 10,
        "text_uri": "minio://text",
        "user_id": "user_123",
    }
    mock_db_session.commit.side_effect = Exception("DB Insert failed")

    with (
        patch("app.workers.tasks.update_job_status"),
        patch(
            "app.workers.tasks.storage_service.download_file_sync",
            return_value=b"text bytes",
        ),
        patch("app.workers.tasks.parser_service.split_text", return_value=["chunk"]),
        patch("app.workers.tasks.ensure_user_partition"),
    ):
        with pytest.raises(Exception, match="DB Insert failed"):
            chunk_task(data)


def test_the_sweeper_logic(mock_db_session):
    """Verify The Sweeper identifies and fails stale jobs."""
    from app.workers.beat_tasks import the_sweeper

    # Using spec=IngestionJob
    mock_job = MagicMock(spec=IngestionJob)
    mock_job.status = JobStatus.PENDING

    mock_db_session.query.return_value.filter.return_value.all.return_value = [mock_job]
    assert the_sweeper() == 1
    assert mock_job.status == JobStatus.FAILED


def test_vector_index_maintenance_success(mock_db_session):
    """Verify vector index maintenance executes with AUTOCOMMIT."""
    from app.workers.beat_tasks import vector_index_maintenance

    with patch("app.workers.beat_tasks.engine") as mock_engine:
        mock_conn_options = (
            mock_engine.connect.return_value.execution_options.return_value
        )
        mock_conn = mock_conn_options.__enter__.return_value
        result = vector_index_maintenance()

        assert result is True
        mock_engine.connect.assert_called_once()
        # Verify isolation level requirement
        mock_engine.connect.return_value.execution_options.assert_called_with(
            isolation_level="AUTOCOMMIT"
        )
        mock_conn.execute.assert_called_once()


def test_vector_index_maintenance_failure(mock_db_session):
    """Verify vector index maintenance handles SQL errors gracefully."""
    from app.workers.beat_tasks import vector_index_maintenance

    with patch("app.workers.beat_tasks.engine") as mock_engine:
        mock_engine.connect.side_effect = Exception("DB Maintenance error")
        result = vector_index_maintenance()
        assert result is False


def test_start_ingestion_pipeline():
    """Verify the pipeline orchestrator creates a Celery chain with ingestion queue."""
    with patch("app.workers.tasks.chain") as mock_chain:
        start_ingestion_pipeline(1, 10)
        mock_chain.assert_called_once()
        mock_chain.return_value.apply_async.assert_called_with(queue="ingestion")


def test_embed_task_mismatched_vectors(mock_db_session):
    """Verify embed_task triggers retry if API returns fewer vectors than chunks."""
    data = {"job_id": 1, "document_id": 10, "user_id": "user_123"}
    chunk1 = MagicMock(spec=DocumentChunk)
    chunk1.text_content = "text 1"

    # Mock the chain of filters for the query
    mock_query = mock_db_session.query.return_value
    mock_query.filter.return_value.filter.return_value.all.return_value = [chunk1]

    mock_llm = MagicMock()
    # API returns 0 vectors for 1 chunk
    mock_llm.get_embeddings_batch_sync.return_value = []

    with (
        patch("app.workers.tasks.get_llm_service", return_value=mock_llm),
        patch("app.workers.tasks.update_job_status"),
        patch.object(embed_task, "retry", side_effect=Exception("Retry Triggered")),
    ):
        # We call the task directly since we're patching the
        # task object's retry method.
        with pytest.raises(Exception, match="Retry Triggered"):
            embed_task(data)

        mock_llm.get_embeddings_batch_sync.assert_called_once()
