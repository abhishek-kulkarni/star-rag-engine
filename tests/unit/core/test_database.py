from unittest.mock import MagicMock

import pytest
from sqlalchemy.orm import Session

from app.core.database import ensure_user_partition, get_db


def test_get_db_yields_session():
    db_gen = get_db()
    db = next(db_gen)
    assert isinstance(db, Session)
    # Cleanup
    try:
        next(db_gen)
    except StopIteration:
        pass


def test_ensure_user_partition_valid():
    """Verify partition creation SQL for valid user_id."""
    db = MagicMock(spec=Session)
    user_id = "user-123"

    ensure_user_partition(db, user_id)

    # Verify SQL execution
    db.execute.assert_called_once()
    args, kwargs = db.execute.call_args
    sql = args[0].text
    params = args[1]

    assert "document_chunks_user_123" in sql
    assert "FOR VALUES IN (:uid)" in sql
    assert params["uid"] == user_id
    db.commit.assert_called_once()


def test_ensure_user_partition_invalid():
    """Verify ValueError is raised for invalid user_id."""
    db = MagicMock(spec=Session)
    invalid_id = "user; DROP TABLE"

    with pytest.raises(ValueError, match="Invalid user_id"):
        ensure_user_partition(db, invalid_id)

    db.execute.assert_not_called()
    db.commit.assert_not_called()
