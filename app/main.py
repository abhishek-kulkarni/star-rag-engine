from fastapi import FastAPI

from app.api.v1.endpoints import documents, query

app = FastAPI(title="STAR RAG Engine")


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


@app.get("/")
async def root():
    return {"message": "Welcome to STAR RAG Engine"}


app.include_router(documents.router, prefix="/api/v1/documents", tags=["documents"])
app.include_router(query.router, prefix="/api/v1/query", tags=["query"])
