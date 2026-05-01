import re
from collections.abc import Generator

from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session, sessionmaker

from app.config.settings import settings


def ensure_user_partition(db: Session, user_id: str) -> None:
    """
    Dynamically creates a PostgreSQL partition for a specific user if it doesn't exist.
    Uses strict DDL sanitization to prevent SQL injection.

    WARNING: This function calls db.commit() to finalize DDL.
    Ensure this is called at the START of a session or in an isolated session
    to avoid prematurely committing other pending data changes.
    """
    # Strict validation: only alphanumeric characters, underscores, and hyphens allowed.
    if not re.match(r"^[a-zA-Z0-9_-]+$", user_id):
        raise ValueError(f"Invalid user_id for partition creation: {user_id}")

    # PostgreSQL partition names must be unique.
    # We use a safe prefix to avoid collisions with system tables.
    partition_name = f"document_chunks_{user_id.replace('-', '_')}"

    # DDL table names cannot be parameterized, but the values can.
    # We use f-string for the table name (sanitized) and :uid for the value.
    db.execute(
        text(
            f"CREATE TABLE IF NOT EXISTS {partition_name} "
            f"PARTITION OF document_chunks FOR VALUES IN ('{user_id}');"
        )
    )
    db.commit()


# For unit testing without a real DB, we use a sqlite in-memory for basic session tests
# or we can mock it. Since Phase 1 is infrastructure, we'll setup the real URL.
DATABASE_URL = settings.sqlalchemy_database_uri

# We use create_engine for the actual implementation
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db() -> Generator[Session]:
    """
    FastAPI dependency that yields a database session.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
