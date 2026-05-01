from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.core.security import get_current_user
from app.models.document import DocumentChunk
from app.models.schemas import RAGQueryRequest, STARAnswerResponse
from app.services.llm_service import llm_service

router = APIRouter()


@router.post("/ask", response_model=STARAnswerResponse)
async def ask_question(
    request: RAGQueryRequest,
    current_user: Annotated[str, Depends(get_current_user)],
    db: Annotated[Session, Depends(get_db)],
):
    """
    Orchestrates the RAG lifecycle:
    1. Embeds the query asynchronously.
    2. Performs similarity search with strict partition pruning (user_id).
    3. Generates a structured STAR answer grounded in the retrieved chunks.
    """
    # 1. Generate query embedding (768-dim)
    try:
        query_vector = await llm_service.get_embeddings(request.query)
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"LLM Embedding service failed: {str(e)}"
        ) from e

    # 2. Similarity search using cosine distance on pgvector
    # MUST filter by user_id to ensure partition pruning and multi-tenant isolation
    chunks = (
        db.query(DocumentChunk)
        .filter(DocumentChunk.user_id == current_user)
        .order_by(DocumentChunk.embedding.cosine_distance(query_vector))
        .limit(5)  # Retrieve top-5 most relevant context segments
        .all()
    )

    if not chunks:
        raise HTTPException(
            status_code=404,
            detail="No relevant context found. Please upload documents first.",
        )

    # 3. Format context for LLM grounding
    retrieved_chunks = [
        {"id": chunk.id, "text_content": chunk.text_content} for chunk in chunks
    ]

    # 4. Generate STAR Answer via Gemini Orchestrator
    try:
        answer = await llm_service.generate_star_answer(
            query=request.query,
            retrieved_chunks=retrieved_chunks,
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"LLM Generation service failed: {str(e)}"
        ) from e

    return answer
