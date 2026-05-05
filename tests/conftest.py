from unittest.mock import patch

import pytest

# --- GLOBAL SAFETY NET: Initialization Only ---
# We surgically mock the bucket check during StorageService initialization
# to prevent collection errors while Docker is down.
patch("app.services.storage_service.StorageService._ensure_bucket_exists").start()


@pytest.fixture(autouse=True)
def protect_llm(request):
    """
    Global safety net to protect Gemini quotas.
    Skips the global mock for LLM unit tests to avoid AsyncMock conflicts.
    """
    if "test_llm_service" in request.node.nodeid:
        yield
    else:
        with patch("google.genai.Client"):
            yield


from sqlalchemy import create_engine  # noqa: E402
from sqlalchemy.orm import sessionmaker  # noqa: E402


@pytest.fixture(scope="session", autouse=True)
def api_safety_net():
    """
    Cleans up global mocks after the test session.
    """
    yield


# Mock pgvector for SQLite compatibility during tests
try:
    import pgvector.sqlalchemy  # noqa: F401
except ImportError:
    pass


# Force SQLite to treat Vector as a plain text or blob for testing
@pytest.fixture(scope="session", autouse=True)
def mock_pgvector_type():
    from sqlalchemy import String

    import app.models.document as doc_module

    # Temporarily replace Vector with String for SQLite tests
    original_vector = doc_module.Vector
    doc_module.Vector = lambda dim: String(dim)
    yield
    doc_module.Vector = original_vector


# Use SQLite for testing (note: pgvector features will be mocked if used)
TEST_SQLALCHEMY_DATABASE_URL = "sqlite:///./test.db"

engine = create_engine(
    TEST_SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


@pytest.fixture(scope="session", autouse=True)
def setup_test_db():
    # Base.metadata.create_all(bind=engine)
    yield
    # Base.metadata.drop_all(bind=engine)


@pytest.fixture
def db_session():
    """Provides a clean database session for each test."""
    connection = engine.connect()
    transaction = connection.begin()
    session = TestingSessionLocal(bind=connection)

    yield session

    session.close()
    transaction.rollback()
    connection.close()
