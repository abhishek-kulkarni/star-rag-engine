from app.services.parser_service import parser_service
from app.workers.celery_app import celery_app


@celery_app.task(name="app.workers.tasks.parse_task")
def parse_task(job_id: str, content: bytes):
    """Minimal implementation of parse_task."""
    return parser_service.parse_pdf(content)


@celery_app.task(name="app.workers.tasks.chunk_task")
def chunk_task(job_id: str, text: str):
    """Minimal implementation of chunk_task."""
    return parser_service.split_text(text)


@celery_app.task(name="app.workers.tasks.embed_task")
def embed_task(job_id: str, chunks: list[str]):
    """Minimal implementation of embed_task."""
    # Placeholder for LLM service integration
    return []
