import contextlib

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from prometheus_client import make_asgi_app
from sqlalchemy import text

from app.api.v1.endpoints import documents, query
from app.config.settings import settings
from app.core.database import engine
from app.core.logging import setup_logging
from app.core.security import get_current_user
from app.models.base import Base

# Ensure models are imported so Base.metadata knows about them

# 1. Initialize Structured Logging
setup_logging()


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup logic to ensure database extensions and tables exist.
    Runs once before the server accepts requests.
    """
    # 1. Ensure the pgvector extension exists
    with engine.begin() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))

    # 2. Automatically create tables if they don't exist
    Base.metadata.create_all(bind=engine)

    yield


app = FastAPI(title="STAR RAG Engine", lifespan=lifespan)

# 2. Enable CORS for browser-based UI interaction
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all for local testing
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 3. Mount Static UI Assets
# The dashboard will be available at http://localhost:8000/ui/
app.mount("/ui", StaticFiles(directory="app/static", html=True), name="static")

# Bypass Auth for local testing
if settings.ENVIRONMENT == "local":
    app.dependency_overrides[get_current_user] = lambda: "eval_user"

# 2. Expose Prometheus Metrics endpoint
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


@app.get("/")
async def root():
    return {"message": "Welcome to STAR RAG Engine"}


app.include_router(documents.router, prefix="/api/v1/documents", tags=["documents"])
app.include_router(query.router, prefix="/api/v1/query", tags=["query"])
