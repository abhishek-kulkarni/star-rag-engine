from pydantic import computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    ENVIRONMENT: str = "local"

    # Database
    POSTGRES_USER: str = "admin"
    POSTGRES_PASSWORD: str = "securepassword"
    POSTGRES_DB: str = "starrag"
    POSTGRES_HOST: str = "localhost"
    POSTGRES_PORT: int = 5432

    # Redis & Celery
    REDIS_URL: str = "redis://redis:6379/0"
    CELERY_DLQ_NAME: str = "starrag_dlq"
    CELERY_INGESTION_QUEUE: str = "ingestion"
    CELERY_DEFAULT_QUEUE: str = "default"

    # MinIO
    MINIO_ENDPOINT: str = "localhost:9000"
    MINIO_ROOT_USER: str = "minioadmin"
    MINIO_ROOT_PASSWORD: str = "minioadmin"
    MINIO_SECURE: bool = False
    MINIO_BUCKET_NAME: str = "starrag-documents"

    # AI & Security
    GEMINI_API_KEY: str = "your_api_key_here"
    PII_CONFIDENCE_THRESHOLD: float = 0.4

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    @computed_field
    def sqlalchemy_database_uri(self) -> str:
        """Dynamically constructs the database connection string for SQLAlchemy."""
        # Note: Swap 'psycopg' with 'asyncpg' if using the async SQLAlchemy engine
        return f"postgresql+psycopg://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"


settings = Settings()
