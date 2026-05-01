from app.config.settings import Settings


def test_settings_load_from_env(monkeypatch):
    monkeypatch.setenv("POSTGRES_USER", "testuser")
    monkeypatch.setenv("GEMINI_API_KEY", "testkey")

    settings = Settings(_env_file=None)
    assert settings.POSTGRES_USER == "testuser"
    assert settings.GEMINI_API_KEY == "testkey"


def test_settings_default_values():
    # Instantiate without loading .env to test hardcoded defaults
    settings = Settings(_env_file=None)
    # Assuming standard defaults from system design
    assert settings.POSTGRES_PORT == 5432
    assert settings.REDIS_URL == "redis://redis:6379/0"
    assert settings.CELERY_DEFAULT_QUEUE == "default"
    assert settings.CELERY_INGESTION_QUEUE == "ingestion"
    assert "postgresql+psycopg://" in settings.sqlalchemy_database_uri
