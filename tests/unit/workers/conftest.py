from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture(autouse=True)
def mock_db_session():
    """
    Globally mocks SessionLocal for all worker tests to prevent real DB connections.
    Uses 'autouse=True' to ensure it's always active in this directory.
    """
    with (
        patch("app.workers.tasks.SessionLocal") as mock_tasks,
        patch("app.workers.beat_tasks.SessionLocal") as mock_beat,
        patch("app.workers.tasks.LLMService") as mock_llm_class,
        patch("app.workers.tasks.storage_service") as mock_storage,
    ):
        session = MagicMock()
        mock_tasks.return_value.__enter__.return_value = session
        mock_beat.return_value.__enter__.return_value = session

        # Force async services to use regular MagicMock to avoid coroutine warnings
        mock_llm_instance = mock_llm_class.return_value
        mock_llm_instance.get_embeddings = MagicMock(return_value=[0.0] * 768)

        mock_storage.download_file = MagicMock(return_value=b"test data")
        mock_storage.upload_file = MagicMock(return_value="minio://test")

        # Mocking the query chain with specs to prevent delusional mocks
        from app.models.document import IngestionJob

        session.query.return_value.filter.return_value.first.return_value = MagicMock(
            spec=IngestionJob
        )
        session.query.return_value.filter.return_value.all.return_value = []

        yield session
