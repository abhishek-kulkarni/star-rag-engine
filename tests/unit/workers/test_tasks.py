from unittest.mock import MagicMock, patch

import pytest

from app.models.document import JobStatus
from app.workers.tasks import (
    chunk_task,
    embed_task,
    get_llm_service,
    parse_task,
    start_ingestion_pipeline,
)


def test_get_llm_service_lazy_init(monkeypatch):
    """Verify LLM service lazy initialization."""
    with patch("app.workers.tasks.LLMService") as mock_class:
        # Manually clear the global to force init
        import app.workers.tasks

        monkeypatch.setattr(app.workers.tasks, "_llm_service", None)
        service = get_llm_service()
        assert service is not None
        mock_class.assert_called_once()


def test_update_job_status_branches(mock_db_session):
    """Verify all branches of update_job_status (timestamps)."""
    from app.workers.tasks import update_job_status

    mock_job = MagicMock()
    mock_db_session.query.return_value.filter.return_value.first.return_value = mock_job

    update_job_status(1, JobStatus.PARSING)
    assert mock_job.status == JobStatus.PARSING

    update_job_status(1, JobStatus.CHUNKING)
    assert mock_job.status == JobStatus.CHUNKING

    update_job_status(1, JobStatus.COMPLETED)
    assert mock_job.status == JobStatus.COMPLETED

    # Test error message branch (Line 32)
    update_job_status(1, JobStatus.FAILED, error="Some error")
    assert mock_job.error_message == "Some error"


def test_parse_task_doc_not_found(mock_db_session):
    """Verify parse_task handles missing document (Line 55)."""
    data = {"job_id": 1, "document_id": 999}
    # Mock Document retrieval to return None
    mock_db_session.query.return_value.filter.return_value.first.return_value = None

    with patch("app.workers.tasks.update_job_status"):
        with pytest.raises(ValueError, match="Document 999 not found"):
            parse_task(data)


def test_parse_task_logic(mock_db_session):
    """Verify parse_task downloads PDF, extracts text, and uploads it."""
    data = {"job_id": 1, "document_id": 10}
    mock_doc = MagicMock(id=10, owner_id=99, storage_uri="minio://pdf")
    mock_db_session.query.return_value.filter.return_value.first.return_value = mock_doc

    with (
        patch("app.workers.tasks.storage_service.download_file"),
        patch("app.workers.tasks.parser_service.parse_pdf") as mock_parse,
        patch("app.workers.tasks.storage_service.upload_file"),
        patch("app.workers.tasks.asyncio.run") as mock_run,
    ):
        mock_run.side_effect = [b"pdf bytes", "minio://text"]
        mock_parse.return_value = "extracted text"
        result = parse_task(data)
        assert result["text_uri"] == "minio://text"
        mock_db_session.commit.assert_called()


def test_chunk_task_logic(mock_db_session):
    """Verify chunk_task downloads text, splits it, and saves to DB."""
    data = {"job_id": 1, "document_id": 10, "text_uri": "minio://text"}
    with (
        patch("app.workers.tasks.storage_service.download_file"),
        patch("app.workers.tasks.parser_service.split_text") as mock_split,
        patch("app.workers.tasks.asyncio.run") as mock_run,
    ):
        mock_run.return_value = b"text bytes"
        mock_split.return_value = ["chunk 1", "chunk 2"]
        result = chunk_task(data)
        assert result["document_id"] == 10
        assert mock_db_session.add.call_count == 2
        mock_db_session.commit.assert_called()


def test_embed_task_logic(mock_db_session):
    """Verify embed_task fetches null-embedding chunks and fills them."""
    data = {"job_id": 1, "document_id": 10}
    chunk1 = MagicMock(text_content="text 1", embedding=None)
    mock_db_session.query.return_value.filter.return_value.all.return_value = [chunk1]

    mock_llm = MagicMock()
    mock_llm.get_embeddings.return_value = [0.1]

    with (
        patch("app.workers.tasks.get_llm_service", return_value=mock_llm),
        patch("app.workers.tasks.asyncio.run", side_effect=lambda x: [0.1]),
    ):
        result = embed_task(data)
        assert result is True
        assert chunk1.embedding == [0.1]
        mock_db_session.commit.assert_called()


def test_parse_task_failure(mock_db_session):
    """Verify parse_task handles errors and updates status."""
    data = {"job_id": 1, "document_id": 10}
    # Mock update_job_status to avoid it failing on the DB mock
    with patch("app.workers.tasks.update_job_status"):
        mock_db_session.query.return_value.filter.return_value.first.side_effect = (
            Exception("DB error")
        )
        with pytest.raises(Exception, match="DB error"):
            parse_task(data)


def test_chunk_task_failure(mock_db_session):
    """Verify chunk_task handles storage errors."""
    data = {"job_id": 1, "document_id": 10, "text_uri": "minio://text"}
    with patch("app.workers.tasks.update_job_status"):
        with patch(
            "app.workers.tasks.asyncio.run", side_effect=Exception("Storage error")
        ):
            with pytest.raises(Exception, match="Storage error"):
                chunk_task(data)


def test_embed_task_failure(mock_db_session):
    """Verify embed_task handles LLM errors."""
    data = {"job_id": 1, "document_id": 10}
    mock_db_session.query.return_value.filter.return_value.all.return_value = [
        MagicMock()
    ]
    with patch("app.workers.tasks.update_job_status"):
        with patch("app.workers.tasks.asyncio.run", side_effect=Exception("LLM error")):
            with pytest.raises(Exception, match="LLM error"):
                embed_task(data)


def test_the_sweeper_logic(mock_db_session):
    """Verify The Sweeper identifies and fails stale jobs."""
    from app.workers.beat_tasks import the_sweeper

    mock_job = MagicMock(status=JobStatus.PENDING)
    mock_db_session.query.return_value.filter.return_value.all.return_value = [mock_job]
    assert the_sweeper() == 1
    assert mock_job.status == JobStatus.FAILED


def test_start_ingestion_pipeline():
    """Verify the pipeline orchestrator creates a Celery chain."""
    with patch("app.workers.tasks.chain") as mock_chain:
        start_ingestion_pipeline(1, 10)
        mock_chain.assert_called_once()
