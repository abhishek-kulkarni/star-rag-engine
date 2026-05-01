from celery import Celery  # type: ignore

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
    # Reliability: Task acknowledgment and retries
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    # Queue Configuration
    task_default_queue="default",
    task_queues={
        "default": {"exchange": "default", "routing_key": "default"},
        "ingestion": {"exchange": "ingestion", "routing_key": "ingestion"},
        "dead_letter": {"exchange": "dead_letter", "routing_key": "dead_letter"},
    },
)

# Periodic Task Scheduling (Celery Beat)
celery_app.conf.beat_schedule = {
    "the-sweeper-every-30-mins": {
        "task": "app.workers.beat_tasks.the_sweeper",
        "schedule": 1800.0,  # 30 minutes
    },
    "vector-index-maintenance-weekly": {
        "task": "app.workers.beat_tasks.vector_index_maintenance",
        "schedule": 604800.0,  # 7 days
    },
}
