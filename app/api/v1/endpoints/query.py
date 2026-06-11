import logging
import time
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.core.logging import telemetry
from app.core.security import get_current_user
from app.models.document import Document, DocumentChunk, DocumentType
from app.models.schemas import QueryResponse, RAGQueryRequest
from app.services.llm_service import llm_service
from app.services.storage_service import storage_service

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
            status_code=500,
            detail="LLM Embedding service failed. Please try again later.",
        ) from e

    # 2. Similarity search using cosine distance on pgvector
    # MUST filter by user_id to ensure partition pruning and multi-tenant isolation
    try:
        chunks = (
            db
            .query(DocumentChunk)
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
            status_code=500, detail="Database search failed. Please try again later."
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

    # 4. Retrieve and Inject Plan Artifacts (Grounding Instructions)

    plan_artifacts = (
        db
        .query(Document)
        .filter(
            Document.user_id == current_user,
            Document.doc_type == DocumentType.PLAN_ARTIFACT,
        )
        .all()
    )

    plan_text = ""
    if plan_artifacts:
        logger.info(
            f"Injecting {len(plan_artifacts)} plan artifacts for user {current_user}"
        )
        # Download and concatenate artifact contents (txt only for now for simplicity)
        # Note: In production, we'd cache these or use a vector store if they get large
        contents = []
        for artifact in plan_artifacts:
            try:
                # We assume plan artifacts are small enough to fit in memory
                content = await storage_service.download_file(artifact.minio_raw_uri)
                contents.append(content.decode("utf-8"))
            except Exception as e:
                logger.warning(
                    f"Failed to load plan artifact {artifact.filename}: {str(e)}"
                )

        plan_text = "\n---\n".join(contents)

    # 5. Generate STAR Answer via Gemini Orchestrator
    try:
        answer = await llm_service.generate_star_answer(
            query=request.query,
            retrieved_chunks=retrieved_chunks,
            plan_artifacts_text=plan_text if plan_text else None,
        )

        # 6. Track Telemetry (Latency)
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
