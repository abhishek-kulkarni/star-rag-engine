from unittest.mock import patch

import pytest
from prometheus_client import CollectorRegistry

from app.core.logging import Telemetry, setup_logging


@pytest.fixture
def registry():
    return CollectorRegistry()


def test_telemetry_metrics_initialization(registry):
    """Verify Prometheus metrics are initialized."""
    telemetry = Telemetry(registry=registry)
    assert hasattr(telemetry, "query_latency")
    assert hasattr(telemetry, "token_usage")
    assert hasattr(telemetry, "storage_errors")
    assert hasattr(telemetry, "llm_errors")


def test_telemetry_track_llm_errors(registry):
    """Verify LLM error tracking with labels."""
    telemetry = Telemetry(registry=registry)
    with patch.object(telemetry.llm_errors, "labels") as mock_labels:
        telemetry.llm_errors.labels(model="generate").inc()
        mock_labels.assert_called_once_with(model="generate")
        mock_labels.return_value.inc.assert_called_once()


def test_telemetry_track_query_latency(registry):
    """Verify latency tracking increment."""
    telemetry = Telemetry(registry=registry)
    with patch.object(telemetry.query_latency, "observe") as mock_observe:
        telemetry.track_query(duration=1.5)
        mock_observe.assert_called_once_with(1.5)


def test_telemetry_track_token_usage(registry):
    """Verify token usage tracking with labels."""
    telemetry = Telemetry(registry=registry)
    with patch.object(telemetry.token_usage, "labels") as mock_labels:
        telemetry.track_tokens(prompt=100, completion=50)
        # Verify both labels are called
        mock_labels.assert_any_call(type="prompt")
        mock_labels.assert_any_call(type="completion")
        # Verify increment
        assert mock_labels.return_value.inc.call_count == 2


@patch("logging.config.dictConfig")
def test_setup_logging_production(mock_dict_config):
    """Verify JSON formatting is applied in production."""
    with patch("app.core.logging.settings") as mock_settings:
        mock_settings.ENVIRONMENT = "production"
        setup_logging()
        # Verify JSON formatter is in the config
        args, _ = mock_dict_config.call_args
        config = args[0]
        assert "json" in config["formatters"]
        assert config["handlers"]["console"]["formatter"] == "json"
