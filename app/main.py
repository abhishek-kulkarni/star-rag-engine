from fastapi import FastAPI

app = FastAPI(title="STAR RAG Engine API")


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


@app.get("/")
async def root():
    return {"message": "Welcome to STAR RAG Engine API"}
