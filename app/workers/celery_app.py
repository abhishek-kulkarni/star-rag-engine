from celery import Celery

from app.config.settings import settings

# Initialize Celery with Redis broker and backend
celery_app = Celery(
    "star_rag_workers",
    broker=settings.REDIS_URL,
    backend=settings.REDIS_URL,
    include=["app.workers.tasks", "app.workers.beat_tasks"],
)

# Principal-level configuration for reliability
celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    # Dead Letter Queue (DLQ) strategy for failing tasks
    task_acks_late=True,
    task_reject_on_worker_lost=True,
)
