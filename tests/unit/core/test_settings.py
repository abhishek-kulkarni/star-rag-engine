from app.config.settings import Settings


def test_settings_load_from_env(monkeypatch):
    monkeypatch.setenv("POSTGRES_USER", "testuser")
    monkeypatch.setenv("GEMINI_API_KEY", "testkey")

    settings = Settings()
    assert settings.POSTGRES_USER == "testuser"
    assert settings.GEMINI_API_KEY == "testkey"


def test_settings_default_values():
    settings = Settings()
    # Assuming standard defaults from system design
    assert settings.POSTGRES_PORT == 5432
    assert settings.REDIS_URL == "redis://redis:6379/0"
