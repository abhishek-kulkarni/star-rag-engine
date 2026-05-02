import logging
import time
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.core.logging import telemetry
from app.core.security import get_current_user
from app.models.document import DocumentChunk
from app.models.schemas import QueryResponse, RAGQueryRequest
from app.services.llm_service import llm_service

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/ask", response_model=QueryResponse)
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
    logger.info(f"Received query from user {current_user}: {request.query[:50]}...")
    start_time = time.perf_counter()

    # 1. Generate query embedding (768-dim)
    try:
        query_vector = await llm_service.get_embeddings(request.query)
    except Exception as e:
        logger.error(f"Embedding failed for user {current_user}: {str(e)}")
        telemetry.llm_errors.labels(model="embed").inc()
        raise HTTPException(
            status_code=500, detail=f"LLM Embedding service failed: {str(e)}"
        ) from e

    # 2. Similarity search using cosine distance on pgvector
    # MUST filter by user_id to ensure partition pruning and multi-tenant isolation
    try:
        chunks = (
            db.query(DocumentChunk)
            .filter(DocumentChunk.user_id == current_user)
            .order_by(DocumentChunk.embedding.cosine_distance(query_vector))
            .limit(5)  # Retrieve top-5 most relevant context segments
            .all()
        )
        logger.info(f"Retrieved {len(chunks)} context chunks for user {current_user}")
    except Exception as e:
        logger.error(f"Similarity search failed for user {current_user}: {str(e)}")
        telemetry.storage_errors.labels(service="postgres").inc()
        raise HTTPException(
            status_code=500, detail=f"Database search failed: {str(e)}"
        ) from e

    if not chunks:
        logger.warning(f"No context found for user {current_user} query")
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

        # 5. Track Telemetry (Latency)
        duration = time.perf_counter() - start_time
        logger.info(f"STAR Answer generated for {current_user} in {duration:.2f}s")
        telemetry.track_query(duration)

        return QueryResponse(answer=answer, source_nodes=retrieved_chunks)
    except Exception as e:
        logger.error(f"Generation failed for user {current_user}: {str(e)}")
        telemetry.llm_errors.labels(model="generate").inc()
        raise HTTPException(
            status_code=500,
            detail="LLM Generation service failed. Please try again later.",
        ) from e
