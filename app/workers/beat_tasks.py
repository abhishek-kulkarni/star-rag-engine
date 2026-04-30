from app.workers.celery_app import celery_app


@celery_app.task(name="app.workers.beat_tasks.the_sweeper")
def the_sweeper():
    """Periodic task for cleaning up stale ingestion jobs."""
    pass
