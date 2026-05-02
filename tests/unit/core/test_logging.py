from unittest.mock import patch

from app.core.logging import telemetry


def test_start_telemetry_server():
    """Verify that the telemetry server starts correctly."""
    with patch("app.core.logging.start_http_server") as mock_start:
        telemetry.start_telemetry_server(port=9999)
        mock_start.assert_called_once_with(9999)


def test_track_tokens():
    """Verify token tracking updates the counter."""
    # We use a fresh registry to avoid global state pollution
    from prometheus_client import CollectorRegistry

    from app.core.logging import Telemetry

    registry = CollectorRegistry()
    test_telemetry = Telemetry(registry=registry)

    test_telemetry.track_tokens(prompt=10, completion=20)

    # Check the value in the registry
    prompt_val = registry.get_sample_value("rag_token_usage_total", {"type": "prompt"})
    completion_val = registry.get_sample_value(
        "rag_token_usage_total", {"type": "completion"}
    )

    assert prompt_val == 10
    assert completion_val == 20


def test_track_query():
    """Verify query latency tracking."""
    from prometheus_client import CollectorRegistry

    from app.core.logging import Telemetry

    registry = CollectorRegistry()
    test_telemetry = Telemetry(registry=registry)
    test_telemetry.track_query(0.5)

    # Check the histogram sample
    val = registry.get_sample_value("rag_query_latency_seconds_sum")
    assert val == 0.5


def test_setup_logging():
    """Verify logging setup runs without error."""
    from app.core.logging import setup_logging

    # Just ensure it doesn't crash
    setup_logging()
