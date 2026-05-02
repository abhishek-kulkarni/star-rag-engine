from unittest.mock import patch

from app.workers.celery_app import setup_direct_metrics


def test_worker_ready_signal_triggers_telemetry():
    """Verify that the worker_ready signal starts the telemetry server."""
    with patch("app.workers.celery_app.telemetry") as mock_telemetry:
        # Manually trigger the signal handler
        setup_direct_metrics(sender=None)

        # Verify the telemetry server was started on the correct port
        mock_telemetry.start_telemetry_server.assert_called_once_with(port=8001)


def test_celery_app_configuration():
    """Verify core Celery configuration settings."""
    from app.workers.celery_app import celery_app

    assert celery_app.conf.task_acks_late is True
    assert "app.workers.tasks" in celery_app.conf.include
    assert celery_app.conf.timezone == "UTC"
