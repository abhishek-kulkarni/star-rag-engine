from unittest.mock import patch

from fastapi.testclient import TestClient

from app.main import app


def test_root_endpoint():
    """Verify the root welcome message."""
    client = TestClient(app)
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Welcome to STAR RAG Engine"}


def test_health_check_endpoint():
    """Verify the health check returns healthy status."""
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_lifespan_initialization():
    """
    Verify that the lifespan context manager triggers DB initialization.
    We mock the engine and Base to ensure they are called.
    """
    with (
        patch("app.main.engine") as mock_engine,
        patch("app.main.Base") as mock_base,
    ):
        # The context manager trigger happens here
        with TestClient(app):
            pass

        # Verify pgvector extension creation
        assert mock_engine.begin.called
        # Verify table creation
        mock_base.metadata.create_all.assert_called_once()
