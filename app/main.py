from fastapi import FastAPI
from prometheus_client import make_asgi_app

from app.api.v1.endpoints import documents, query
from app.core.logging import setup_logging

# 1. Initialize Structured Logging
setup_logging()

app = FastAPI(title="STAR RAG Engine")

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
