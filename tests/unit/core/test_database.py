from app.core.database import get_db
from sqlalchemy.orm import Session


def test_get_db_yields_session():
    db_gen = get_db()
    db = next(db_gen)
    assert isinstance(db, Session)
    # Cleanup
    try:
        next(db_gen)
    except StopIteration:
        pass
