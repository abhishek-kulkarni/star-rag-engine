from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.core.security import get_current_user
from app.main import app
from app.models.document import DocumentChunk
from app.models.schemas import STARAnswerResponse

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


def test_ask_question_success(mock_db):
    """Verify successful RAG query flow."""
    with (
        patch(
            "app.api.v1.endpoints.query.llm_service.get_embeddings",
            new_callable=AsyncMock,
        ) as mock_embed,
        patch(
            "app.api.v1.endpoints.query.llm_service.generate_star_answer",
            new_callable=AsyncMock,
        ) as mock_gen,
    ):
        mock_embed.return_value = [0.1] * 768

        # Mock retrieved chunks
        mock_chunk = MagicMock(spec=DocumentChunk)
        mock_chunk.id = 1
        mock_chunk.text_content = "Retrieved context"

        query_mock = mock_db.query.return_value
        (
            query_mock.filter.return_value.order_by.return_value.limit.return_value.all.return_value
        ) = [mock_chunk]

        mock_gen.return_value = STARAnswerResponse(
            situation="S", task="T", action="A", result="R", citations=[1]
        )

        response = client.post("/api/v1/query/ask", json={"query": "test query"})

        assert response.status_code == 200
        data = response.json()
        assert data["answer"]["situation"] == "S"
        assert len(data["source_nodes"]) == 1
        assert data["source_nodes"][0]["text_content"] == "Retrieved context"
        mock_embed.assert_called_once_with("test query")
        mock_gen.assert_called_once()


def test_ask_question_embedding_failure(mock_db):
    """Verify 500 error when embedding service fails."""
    with patch(
        "app.api.v1.endpoints.query.llm_service.get_embeddings",
        side_effect=Exception("API Key expired"),
    ):
        response = client.post("/api/v1/query/ask", json={"query": "test query"})
        assert response.status_code == 500
        assert "Embedding service failed" in response.json()["detail"]


def test_ask_question_no_context(mock_db):
    """Verify 404 when no relevant documents are found."""
    with patch(
        "app.api.v1.endpoints.query.llm_service.get_embeddings",
        new_callable=AsyncMock,
    ) as mock_embed:
        mock_embed.return_value = [0.1] * 768

        # Mock empty DB result
        query_mock = mock_db.query.return_value
        (
            query_mock.filter.return_value.order_by.return_value.limit.return_value.all.return_value
        ) = []

        response = client.post("/api/v1/query/ask", json={"query": "missing query"})
        assert response.status_code == 404
        assert "No relevant context" in response.json()["detail"]


def test_ask_question_generation_failure(mock_db):
    """Verify 500 error when LLM generation fails."""
    with (
        patch(
            "app.api.v1.endpoints.query.llm_service.get_embeddings",
            new_callable=AsyncMock,
        ) as mock_embed,
        patch(
            "app.api.v1.endpoints.query.llm_service.generate_star_answer",
            side_effect=Exception("Quota exceeded"),
        ),
    ):
        mock_embed.return_value = [0.1] * 768

        # Mock successful retrieval
        mock_chunk = MagicMock(spec=DocumentChunk)
        mock_chunk.id = 1
        mock_chunk.text_content = "Context"
        query_mock = mock_db.query.return_value
        (
            query_mock.filter.return_value.order_by.return_value.limit.return_value.all.return_value
        ) = [mock_chunk]

        response = client.post("/api/v1/query/ask", json={"query": "test query"})
        assert response.status_code == 500
        assert "Generation service failed" in response.json()["detail"]


def test_ask_question_database_failure(mock_db):
    """Verify 500 error when database search fails."""
    with patch(
        "app.api.v1.endpoints.query.llm_service.get_embeddings",
        new_callable=AsyncMock,
    ) as mock_embed:
        mock_embed.return_value = [0.1] * 768

        # Mock database exception
        mock_db.query.side_effect = Exception("Connection lost")

        response = client.post("/api/v1/query/ask", json={"query": "test query"})
        assert response.status_code == 500
        assert "Database search failed" in response.json()["detail"]


def test_metrics_endpoint():
    """Verify Prometheus metrics are exposed."""
    response = client.get("/metrics/")
    assert response.status_code == 200
    assert "rag_query_latency_seconds" in response.text


def test_ask_question_with_plan_artifact_success(mock_db):
    """
    Verify that plan artifact content is downloaded and passed to the LLM.
    Covers lines 99-102 (the try block inside the artifact download loop).
    """
    from app.models.document import Document

    with (
        patch(
            "app.api.v1.endpoints.query.llm_service.get_embeddings",
            new_callable=AsyncMock,
        ) as mock_embed,
        patch(
            "app.api.v1.endpoints.query.llm_service.generate_star_answer",
            new_callable=AsyncMock,
        ) as mock_gen,
        patch(
            "app.api.v1.endpoints.query.storage_service.download_file",
            new_callable=AsyncMock,
        ) as mock_download,
    ):
        mock_embed.return_value = [0.1] * 768

        # First query() call → DocumentChunk similarity search
        mock_chunk = MagicMock(spec=DocumentChunk)
        mock_chunk.id = 1
        mock_chunk.text_content = "Retrieved context"

        # Second query() call → Document plan artifact lookup
        mock_artifact = MagicMock(spec=Document)
        mock_artifact.minio_raw_uri = "s3://bucket/user/rubric.txt"
        mock_artifact.filename = "rubric.txt"

        # Wire up the two separate db.query() chains
        chunk_chain = MagicMock()
        chunk_result = chunk_chain.filter.return_value.order_by.return_value
        chunk_result.limit.return_value.all.return_value = [mock_chunk]

        artifact_chain = MagicMock()
        artifact_chain.filter.return_value.filter.return_value.all.return_value = [
            mock_artifact
        ]
        artifact_chain.filter.return_value.all.return_value = [mock_artifact]

        mock_db.query.side_effect = lambda model: (
            chunk_chain if model is DocumentChunk else artifact_chain
        )

        mock_download.return_value = b"Always own the outcome."
        mock_gen.return_value = STARAnswerResponse(
            situation="S", task="T", action="A", result="R", citations=[1]
        )

        response = client.post("/api/v1/query/ask", json={"query": "test query"})

        assert response.status_code == 200
        # Confirm the artifact text was passed through to the LLM
        call_kwargs = mock_gen.call_args.kwargs
        assert call_kwargs["plan_artifacts_text"] == "Always own the outcome."


def test_ask_question_plan_artifact_download_failure(mock_db):
    """
    Verify that a failing artifact download is gracefully swallowed (logged as
    warning) and the query still completes successfully without the artifact text.
    Covers lines 103-106 (the except block inside the artifact download loop).
    """
    from app.models.document import Document

    with (
        patch(
            "app.api.v1.endpoints.query.llm_service.get_embeddings",
            new_callable=AsyncMock,
        ) as mock_embed,
        patch(
            "app.api.v1.endpoints.query.llm_service.generate_star_answer",
            new_callable=AsyncMock,
        ) as mock_gen,
        patch(
            "app.api.v1.endpoints.query.storage_service.download_file",
            new_callable=AsyncMock,
            side_effect=Exception("MinIO unreachable"),
        ),
    ):
        mock_embed.return_value = [0.1] * 768

        mock_chunk = MagicMock(spec=DocumentChunk)
        mock_chunk.id = 1
        mock_chunk.text_content = "Retrieved context"

        mock_artifact = MagicMock(spec=Document)
        mock_artifact.minio_raw_uri = "s3://bucket/user/rubric.txt"
        mock_artifact.filename = "rubric.txt"

        chunk_chain = MagicMock()
        chunk_result = chunk_chain.filter.return_value.order_by.return_value
        chunk_result.limit.return_value.all.return_value = [mock_chunk]

        artifact_chain = MagicMock()
        artifact_chain.filter.return_value.all.return_value = [mock_artifact]

        mock_db.query.side_effect = lambda model: (
            chunk_chain if model is DocumentChunk else artifact_chain
        )

        mock_gen.return_value = STARAnswerResponse(
            situation="S", task="T", action="A", result="R", citations=[1]
        )

        # Query must succeed despite the artifact download error
        response = client.post("/api/v1/query/ask", json={"query": "test query"})

        assert response.status_code == 200
        # No artifact text should have been passed —
        # download failed → empty contents list.
        call_kwargs = mock_gen.call_args.kwargs
        assert call_kwargs["plan_artifacts_text"] is None
