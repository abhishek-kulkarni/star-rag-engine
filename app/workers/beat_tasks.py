from datetime import UTC, datetime, timedelta

from app.core.database import SessionLocal
from app.models.document import IngestionJob, JobStatus
from app.workers.celery_app import celery_app


@celery_app.task(name="app.workers.beat_tasks.the_sweeper")
def the_sweeper():
    """
    Garbage collection for orphaned or timed-out ingestion jobs.
    Any job stuck in a non-final state for > 30 minutes is marked as FAILED.
    """
    timeout_threshold = datetime.now(UTC) - timedelta(minutes=30)

    with SessionLocal() as db:
        stale_jobs = (
            db.query(IngestionJob)
            .filter(
                IngestionJob.status.in_(
                    [
                        JobStatus.PENDING,
                        JobStatus.PARSING,
                        JobStatus.CHUNKING,
                        JobStatus.EMBEDDING,
                    ]
                ),
                IngestionJob.created_at < timeout_threshold,
            )
            .all()
        )

        for job in stale_jobs:
            job.status = JobStatus.FAILED
            job.error_message = "Job timed out (The Sweeper GC)"

        db.commit()
        return len(stale_jobs)
