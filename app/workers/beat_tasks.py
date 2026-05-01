import logging
from datetime import UTC, datetime, timedelta

from sqlalchemy import text

from app.core.database import SessionLocal, engine
from app.core.logging import telemetry
from app.models.document import IngestionJob, JobStatus
from app.workers.celery_app import celery_app

logger = logging.getLogger(__name__)


@celery_app.task(name="app.workers.beat_tasks.the_sweeper")
def the_sweeper():
    """
    Identifies and fails orphaned ingestion jobs that have exceeded
    the 1-hour timeout.
    """
    timeout_threshold = datetime.now(UTC) - timedelta(hours=1)

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

        count = 0
        for job in stale_jobs:
            job.status = JobStatus.FAILED
            job.error_message = "Job timed out (The Sweeper)"
            count += 1

        db.commit()
        if count > 0:
            logger.warning(f"The Sweeper: Marked {count} stale jobs as FAILED.")

        return count


@celery_app.task(name="app.workers.beat_tasks.vector_index_maintenance")
def vector_index_maintenance():
    """
    Performs zero-downtime maintenance on pgvector HNSW indexes.
    Rebuilds the graph to remove tombstones and rebalance edges,
    restoring sub-millisecond similarity search performance.
    """
    # From app/models/document.py: DocumentChunk uses 'idx_document_chunks_embedding'
    index_name = "idx_document_chunks_embedding"

    try:
        # We cannot use SessionLocal() because REINDEX CONCURRENTLY cannot run
        # inside a transaction block. We must use a raw connection with AUTOCOMMIT.
        with engine.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
            logger.info(f"Starting concurrent reindex for {index_name}...")

            # Rebuilds the HNSW graph in the background without locking
            # the table for SELECTs.
            conn.execute(text(f"REINDEX INDEX CONCURRENTLY {index_name};"))

            logger.info(f"Vector Index Maintenance: Successfully rebuilt {index_name}.")
            return True

    except Exception as e:
        telemetry.storage_errors.labels(service="postgres").inc()
        logger.error(f"Vector Index Maintenance failed: {e}")
        return False
